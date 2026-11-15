import java.io.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class Transformer implements Serializable 
{
    private static final long serialVersionUID = 20261115L;

    // Hyperparameters
    public int V;      // Vocabulary size
    public int T;      // Sequence / Context length
    public int D;      // Embedding dimension
    public int nHeads; // Number of attention heads
    public int dk;     // Dimension per head (D / nHeads)
    public int H_mlp;  // Feed-forward hidden dimension
    public int L;      // Number of stacked blocks
    public boolean useLayerNorm;

    // Model Parameters
    public double[][] C;       // [V][D] Token embedding table
    public double[][] P;       // [T][D] Position embedding table

    public static class Block implements Serializable 
    {
        private static final long serialVersionUID = 20261115L;

        // Attention weights: [nHeads][D][dk]
        public double[][][] Wq;
        public double[][][] Wk;
        public double[][][] Wv;
        public double[][] Wo;  // [D][D] Output projection

        // Feed-Forward weights
        public double[][] W1;  // [D][H_mlp]
        public double[] b1;    // [H_mlp]
        public double[][] W2;  // [H_mlp][D]
        public double[] b2;    // [D]

        // LayerNorm parameters
        public double[] gamma1; // [D]
        public double[] beta1;  // [D]
        public double[] gamma2; // [D]
        public double[] beta2;  // [D]
    }

    public Block[] blocks;

    // Final LayerNorm (for Pre-LN) & Output Head
    public double[] gammaF;    // [D]
    public double[] betaF;     // [D]
    public double[][] Wout;    // [D][V]
    public double[] bout;      // [V]

    // =========================================================================
    // OPTIMIZER STATE (Adam first and second moments)
    // =========================================================================
    public static class AdamState implements Serializable 
    {
        private static final long serialVersionUID = 20261115L;

        public long t = 0; // step count

        // Moments for C, P, Wout, bout
        public double[][] mC, vC;
        public double[][] mP, vP;
        public double[][] mWout, vWout;
        public double[] mbout, vbout;
        public double[] mGammaF, vGammaF;
        public double[] mBetaF, vBetaF;

        // Moments per block
        public static class BlockAdam implements Serializable 
        {
            private static final long serialVersionUID = 20261115L;
            public double[][][] mWq, vWq;
            public double[][][] mWk, vWk;
            public double[][][] mWv, vWv;
            public double[][] mWo, vWo;

            public double[][] mW1, vW1;
            public double[] mb1, vb1;
            public double[][] mW2, vW2;
            public double[] mb2, vb2;

            public double[] mGamma1, vGamma1;
            public double[] mBeta1, vBeta1;
            public double[] mGamma2, vGamma2;
            public double[] mBeta2, vBeta2;
        }

        public BlockAdam[] blockAdam;
    }

    public AdamState adam;

    // Accumulated Gradients for Mini-batching
    public transient Gradients accumGrads;
    public transient int accumCount = 0;

    // =========================================================================
    // GRADIENT CONTAINERS
    // =========================================================================
    public static class BlockGradients 
    {
        public double[][][] dWq;
        public double[][][] dWk;
        public double[][][] dWv;
        public double[][] dWo;

        public double[][] dW1;
        public double[] db1;
        public double[][] dW2;
        public double[] db2;

        public double[] dGamma1, dBeta1;
        public double[] dGamma2, dBeta2;

        public BlockGradients(int nHeads, int D, int dk, int H_mlp) 
        {
            dWq = new double[nHeads][D][dk];
            dWk = new double[nHeads][D][dk];
            dWv = new double[nHeads][D][dk];
            dWo = new double[D][D];

            dW1 = new double[D][H_mlp];
            db1 = new double[H_mlp];
            dW2 = new double[H_mlp][D];
            db2 = new double[D];

            dGamma1 = new double[D];
            dBeta1 = new double[D];
            dGamma2 = new double[D];
            dBeta2 = new double[D];
        }

        public void zero() 
        {
            zero3D(dWq); zero3D(dWk); zero3D(dWv);
            zero2D(dWo); zero2D(dW1); zero1D(db1); zero2D(dW2); zero1D(db2);
            zero1D(dGamma1); zero1D(dBeta1); zero1D(dGamma2); zero1D(dBeta2);
        }
    }

    public static class Gradients 
    {
        public double[][] dC;
        public double[][] dP;
        public BlockGradients[] blockGrads;
        public double[] dGammaF, dBetaF;
        public double[][] dWout;
        public double[] dbout;

        public Gradients(int V, int T, int D, int nHeads, int dk, int H_mlp, int L) 
        {
            dC = new double[V][D];
            dP = new double[T][D];
            blockGrads = new BlockGradients[L];
            for (int l = 0; l < L; l++) 
            {
                blockGrads[l] = new BlockGradients(nHeads, D, dk, H_mlp);
            }
            dGammaF = new double[D];
            dBetaF = new double[D];
            dWout = new double[D][V];
            dbout = new double[V];
        }

        public void zero() 
        {
            zero2D(dC);
            zero2D(dP);
            for (BlockGradients bg : blockGrads) bg.zero();
            zero1D(dGammaF);
            zero1D(dBetaF);
            zero2D(dWout);
            zero1D(dbout);
        }

        public void add(Gradients other) 
        {
            add2D(dC, other.dC);
            add2D(dP, other.dP);
            for (int l = 0; l < blockGrads.length; l++) 
            {
                BlockGradients bg = blockGrads[l];
                BlockGradients obg = other.blockGrads[l];
                add3D(bg.dWq, obg.dWq);
                add3D(bg.dWk, obg.dWk);
                add3D(bg.dWv, obg.dWv);
                add2D(bg.dWo, obg.dWo);
                add2D(bg.dW1, obg.dW1);
                add1D(bg.db1, obg.db1);
                add2D(bg.dW2, obg.dW2);
                add1D(bg.db2, obg.db2);
                add1D(bg.dGamma1, obg.dGamma1);
                add1D(bg.dBeta1, obg.dBeta1);
                add1D(bg.dGamma2, obg.dGamma2);
                add1D(bg.dBeta2, obg.dBeta2);
            }
            add1D(dGammaF, other.dGammaF);
            add1D(dBetaF, other.dBetaF);
            add2D(dWout, other.dWout);
            add1D(dbout, other.dbout);
        }

        public void scale(double factor) 
        {
            scale2D(dC, factor);
            scale2D(dP, factor);
            for (BlockGradients bg : blockGrads) 
            {
                scale3D(bg.dWq, factor);
                scale3D(bg.dWk, factor);
                scale3D(bg.dWv, factor);
                scale2D(bg.dWo, factor);
                scale2D(bg.dW1, factor);
                scale1D(bg.db1, factor);
                scale2D(bg.dW2, factor);
                scale1D(bg.db2, factor);
                scale1D(bg.dGamma1, factor);
                scale1D(bg.dBeta1, factor);
                scale1D(bg.dGamma2, factor);
                scale1D(bg.dBeta2, factor);
            }
            scale1D(dGammaF, factor);
            scale1D(dBetaF, factor);
            scale2D(dWout, factor);
            scale1D(dbout, factor);
        }
    }

    // =========================================================================
    // INITIALIZATION
    // =========================================================================
    public static Transformer init(int V, int T, int D, int nHeads, int H_mlp, int L, boolean useLayerNorm, double initScale, long seed) 
    {
        Random rng = (seed != 0) ? new Random(seed) : new Random();
        Transformer model = new Transformer();
        model.V = V;
        model.T = T;
        model.D = D;
        model.nHeads = nHeads;
        model.dk = D / nHeads;
        model.H_mlp = H_mlp;
        model.L = L;
        model.useLayerNorm = useLayerNorm;

        model.C = randomMatrix(V, D, initScale, rng);
        model.P = randomMatrix(T, D, initScale, rng);

        model.blocks = new Block[L];
        for (int l = 0; l < L; l++) 
        {
            Block b = new Block();
            b.Wq = new double[nHeads][D][model.dk];
            b.Wk = new double[nHeads][D][model.dk];
            b.Wv = new double[nHeads][D][model.dk];
            for (int h = 0; h < nHeads; h++) 
            {
                b.Wq[h] = randomMatrix(D, model.dk, initScale, rng);
                b.Wk[h] = randomMatrix(D, model.dk, initScale, rng);
                b.Wv[h] = randomMatrix(D, model.dk, initScale, rng);
            }
            b.Wo = randomMatrix(D, D, initScale, rng);

            b.W1 = randomMatrix(D, H_mlp, initScale, rng);
            b.b1 = new double[H_mlp];
            b.W2 = randomMatrix(H_mlp, D, initScale, rng);
            b.b2 = new double[D];

            b.gamma1 = onesVector(D);
            b.beta1 = new double[D];
            b.gamma2 = onesVector(D);
            b.beta2 = new double[D];

            model.blocks[l] = b;
        }

        model.gammaF = onesVector(D);
        model.betaF = new double[D];
        model.Wout = randomMatrix(D, V, initScale, rng);
        model.bout = new double[V];

        model.initAdam();
        model.accumGrads = new Gradients(V, T, D, nHeads, model.dk, H_mlp, L);
        return model;
    }

    public void initAdam() 
    {
        adam = new AdamState();
        adam.mC = new double[V][D]; adam.vC = new double[V][D];
        adam.mP = new double[T][D]; adam.vP = new double[T][D];
        adam.mWout = new double[D][V]; adam.vWout = new double[D][V];
        adam.mbout = new double[V]; adam.vbout = new double[V];
        adam.mGammaF = new double[D]; adam.vGammaF = new double[D];
        adam.mBetaF = new double[D]; adam.vBetaF = new double[D];

        adam.blockAdam = new AdamState.BlockAdam[L];
        for (int l = 0; l < L; l++) 
        {
            AdamState.BlockAdam ba = new AdamState.BlockAdam();
            ba.mWq = new double[nHeads][D][dk]; ba.vWq = new double[nHeads][D][dk];
            ba.mWk = new double[nHeads][D][dk]; ba.vWk = new double[nHeads][D][dk];
            ba.mWv = new double[nHeads][D][dk]; ba.vWv = new double[nHeads][D][dk];
            ba.mWo = new double[D][D]; ba.vWo = new double[D][D];

            ba.mW1 = new double[D][H_mlp]; ba.vW1 = new double[D][H_mlp];
            ba.mb1 = new double[H_mlp]; ba.vb1 = new double[H_mlp];
            ba.mW2 = new double[H_mlp][D]; ba.vW2 = new double[H_mlp][D];
            ba.mb2 = new double[D]; ba.vb2 = new double[D];

            ba.mGamma1 = new double[D]; ba.vGamma1 = new double[D];
            ba.mBeta1 = new double[D]; ba.vBeta1 = new double[D];
            ba.mGamma2 = new double[D]; ba.vGamma2 = new double[D];
            ba.mBeta2 = new double[D]; ba.vBeta2 = new double[D];

            adam.blockAdam[l] = ba;
        }
    }

    public int totalParameters() 
    {
        int total = (V * D) + (T * D); // C + P
        for (Block b : blocks) 
        {
            total += 3 * nHeads * (D * dk); // Wq, Wk, Wv
            total += (D * D);               // Wo
            total += (D * H_mlp) + H_mlp;   // W1 + b1
            total += (H_mlp * D) + D;       // W2 + b2
            if (useLayerNorm) 
            {
                total += 4 * D; // gamma1, beta1, gamma2, beta2
            }
        }
        if (useLayerNorm) 
        {
            total += 2 * D; // gammaF, betaF
        }
        total += (D * V) + V; // Wout + bout
        return total;
    }

    // =========================================================================
    // LAYER NORM UTILITIES
    // =========================================================================
    public static class LNResult 
    {
        public double[] y;
        public double[] xHat;
        public double mean;
        public double var;
        public double stdInv;

        public LNResult(int D) 
        {
            y = new double[D];
            xHat = new double[D];
        }
    }

    public static LNResult layerNormForward(double[] x, double[] gamma, double[] beta, double eps) 
    {
        int D = x.length;
        LNResult res = new LNResult(D);
        double sum = 0.0;
        for (int i = 0; i < D; i++) sum += x[i];
        double mean = sum / D;

        double sumSq = 0.0;
        for (int i = 0; i < D; i++) 
        {
            double diff = x[i] - mean;
            sumSq += diff * diff;
        }
        double var = sumSq / D;
        double stdInv = 1.0 / Math.sqrt(var + eps);

        res.mean = mean;
        res.var = var;
        res.stdInv = stdInv;

        for (int i = 0; i < D; i++) 
        {
            res.xHat[i] = (x[i] - mean) * stdInv;
            res.y[i] = (gamma != null) ? (gamma[i] * res.xHat[i] + beta[i]) : res.xHat[i];
        }
        return res;
    }

    public static void layerNormBackward(double[] dy, LNResult ln, double[] gamma, double[] dx, double[] dGamma, double[] dBeta) 
    {
        int D = dy.length;
        double[] dxHat = new double[D];
        for (int i = 0; i < D; i++) 
        {
            if (dGamma != null) dGamma[i] += dy[i] * ln.xHat[i];
            if (dBeta != null) dBeta[i] += dy[i];
            dxHat[i] = (gamma != null) ? (dy[i] * gamma[i]) : dy[i];
        }

        double sumDxHat = 0.0;
        double sumDxHatXHat = 0.0;
        for (int i = 0; i < D; i++) 
        {
            sumDxHat += dxHat[i];
            sumDxHatXHat += dxHat[i] * ln.xHat[i];
        }

        double invD = 1.0 / D;
        for (int i = 0; i < D; i++) 
        {
            dx[i] = ln.stdInv * (dxHat[i] - invD * sumDxHat - invD * ln.xHat[i] * sumDxHatXHat);
        }
    }

    // =========================================================================
    // FORWARD ACTIVATION CACHE
    // =========================================================================
    public static class BlockCache 
    {
        public double[][] X;           // [T][D] Input to block
        public LNResult[] ln1;         // [T] LN1 cache
        public double[][][] Q;         // [nHeads][T][dk]
        public double[][][] K;         // [nHeads][T][dk]
        public double[][][] V;         // [nHeads][T][dk]
        public double[][][] scores;    // [nHeads][T][T]
        public double[][][] A;         // [nHeads][T][T] Softmax attention matrix
        public double[][] headConcat;  // [T][D] Concat of heads
        public double[][] attOut;      // [T][D] After Wo projection
        public double[][] Y;           // [T][D] After residual: X + attOut

        public LNResult[] ln2;         // [T] LN2 cache
        public double[][] hRaw;        // [T][H_mlp] Before tanh
        public double[][] hTanh;       // [T][H_mlp] After tanh
        public double[][] dropMask;    // [T][H_mlp] Dropout mask
        public double[][] hDrop;       // [T][H_mlp] After dropout
        public double[][] ffOut;       // [T][D] After W2 + b2
        public double[][] Z;           // [T][D] Block output: Y + ffOut

        public BlockCache(int T, int D, int nHeads, int dk, int H_mlp) 
        {
            X = new double[T][D];
            ln1 = new LNResult[T];
            Q = new double[nHeads][T][dk];
            K = new double[nHeads][T][dk];
            V = new double[nHeads][T][dk];
            scores = new double[nHeads][T][T];
            A = new double[nHeads][T][T];
            headConcat = new double[T][D];
            attOut = new double[T][D];
            Y = new double[T][D];

            ln2 = new LNResult[T];
            hRaw = new double[T][H_mlp];
            hTanh = new double[T][H_mlp];
            dropMask = new double[T][H_mlp];
            hDrop = new double[T][H_mlp];
            ffOut = new double[T][D];
            Z = new double[T][D];
        }
    }

    public static class Cache 
    {
        public int[] tokens;
        public int[] targets;
        public double[][] emb;         // [T][D] C + P
        public BlockCache[] blocks;    // [L]
        public LNResult[] lnF;         // [T] Final LN
        public double[][] logits;      // [T][V]
        public double[][] probs;       // [T][V]
        public double[] losses;        // [T] Cross-entropy losses
        public double avgLoss;

        public Cache(int L, int T, int D, int nHeads, int dk, int H_mlp, int V) 
        {
            emb = new double[T][D];
            blocks = new BlockCache[L];
            for (int l = 0; l < L; l++) 
            {
                blocks[l] = new BlockCache(T, D, nHeads, dk, H_mlp);
            }
            lnF = new LNResult[T];
            logits = new double[T][V];
            probs = new double[T][V];
            losses = new double[T];
        }
    }

    // =========================================================================
    // FORWARD PASS
    // =========================================================================
    public Cache forward(int[] tokens, int[] targets, boolean isTraining, double dropoutRate) 
    {
        Cache cache = new Cache(L, T, D, nHeads, dk, H_mlp, V);
        cache.tokens = tokens;
        cache.targets = targets;

        // 1. Embedding layer: C[token] + P[pos]
        for (int t = 0; t < T; t++) 
        {
            int tok = tokens[t];
            for (int d = 0; d < D; d++) 
            {
                cache.emb[t][d] = C[tok][d] + P[t][d];
            }
        }

        // 2. Transformer Blocks
        double[][] curX = cache.emb;
        double sqrtDk = Math.sqrt(dk);

        for (int l = 0; l < L; l++) 
        {
            Block b = blocks[l];
            BlockCache bc = cache.blocks[l];

            // Copy input
            for (int t = 0; t < T; t++) 
                System.arraycopy(curX[t], 0, bc.X[t], 0, D);

            // LayerNorm 1 (Pre-LN)
            double[][] normX1 = bc.X;
            if (useLayerNorm) 
            {
                normX1 = new double[T][D];
                for (int t = 0; t < T; t++) 
                {
                    bc.ln1[t] = layerNormForward(bc.X[t], b.gamma1, b.beta1, 1e-5);
                    normX1[t] = bc.ln1[t].y;
                }
            }

            // Multi-Head Attention: Q, K, V
            for (int h = 0; h < nHeads; h++) 
            {
                for (int t = 0; t < T; t++) 
                {
                    for (int k = 0; k < dk; k++) 
                    {
                        double sumQ = 0.0, sumK = 0.0, sumV = 0.0;
                        for (int d = 0; d < D; d++) 
                        {
                            double xVal = normX1[t][d];
                            sumQ += xVal * b.Wq[h][d][k];
                            sumK += xVal * b.Wk[h][d][k];
                            sumV += xVal * b.Wv[h][d][k];
                        }
                        bc.Q[h][t][k] = sumQ;
                        bc.K[h][t][k] = sumK;
                        bc.V[h][t][k] = sumV;
                    }
                }

                // Causal Attention Scores & Softmax
                for (int t = 0; t < T; t++) 
                {
                    double maxScore = -1e9;
                    for (int i = 0; i < T; i++) 
                    {
                        if (i <= t) 
                        {
                            double dot = 0.0;
                            for (int k = 0; k < dk; k++) dot += bc.Q[h][t][k] * bc.K[h][i][k];
                            bc.scores[h][t][i] = dot / sqrtDk;
                            if (bc.scores[h][t][i] > maxScore) maxScore = bc.scores[h][t][i];
                        } 
                        else 
                        {
                            bc.scores[h][t][i] = -1e9; // Causal future mask
                        }
                    }

                    // Softmax along row i <= t
                    double sumExp = 0.0;
                    for (int i = 0; i <= t; i++) 
                    {
                        bc.A[h][t][i] = Math.exp(bc.scores[h][t][i] - maxScore);
                        sumExp += bc.A[h][t][i];
                    }
                    for (int i = 0; i < T; i++) 
                    {
                        if (i <= t) bc.A[h][t][i] /= sumExp;
                        else bc.A[h][t][i] = 0.0;
                    }

                    // Weighted sum over V
                    for (int k = 0; k < dk; k++) 
                    {
                        double headVal = 0.0;
                        for (int i = 0; i <= t; i++) 
                        {
                            headVal += bc.A[h][t][i] * bc.V[h][i][k];
                        }
                        bc.headConcat[t][h * dk + k] = headVal;
                    }
                }
            }

            // Output projection Wo & Residual Y = X + attOut
            for (int t = 0; t < T; t++) 
            {
                for (int d = 0; d < D; d++) 
                {
                    double sumWo = 0.0;
                    for (int j = 0; j < D; j++) 
                    {
                        sumWo += bc.headConcat[t][j] * b.Wo[j][d];
                    }
                    bc.attOut[t][d] = sumWo;
                    bc.Y[t][d] = bc.X[t][d] + sumWo; // Additive Residual
                }
            }

            // LayerNorm 2 (Pre-LN)
            double[][] normY = bc.Y;
            if (useLayerNorm) 
            {
                normY = new double[T][D];
                for (int t = 0; t < T; t++) 
                {
                    bc.ln2[t] = layerNormForward(bc.Y[t], b.gamma2, b.beta2, 1e-5);
                    normY[t] = bc.ln2[t].y;
                }
            }

            // Position-wise Feed-Forward: W1 -> tanh -> Dropout -> W2
            for (int t = 0; t < T; t++) 
            {
                for (int h = 0; h < H_mlp; h++) 
                {
                    double sum = b.b1[h];
                    for (int d = 0; d < D; d++) sum += normY[t][d] * b.W1[d][h];
                    bc.hRaw[t][h] = sum;
                    bc.hTanh[t][h] = Math.tanh(sum);

                    if (isTraining && dropoutRate > 0.0) 
                    {
                        boolean keep = ThreadLocalRandom.current().nextDouble() >= dropoutRate;
                        bc.dropMask[t][h] = keep ? (1.0 / (1.0 - dropoutRate)) : 0.0;
                        bc.hDrop[t][h] = bc.hTanh[t][h] * bc.dropMask[t][h];
                    } 
                    else 
                    {
                        bc.dropMask[t][h] = 1.0;
                        bc.hDrop[t][h] = bc.hTanh[t][h];
                    }
                }

                for (int d = 0; d < D; d++) 
                {
                    double sum = b.b2[d];
                    for (int h = 0; h < H_mlp; h++) sum += bc.hDrop[t][h] * b.W2[h][d];
                    bc.ffOut[t][d] = sum;
                    bc.Z[t][d] = bc.Y[t][d] + sum; // Additive Residual
                }
            }

            curX = bc.Z;
        }

        // 3. Final LayerNorm (Pre-LN)
        double[][] finalFeatures = curX;
        if (useLayerNorm) 
        {
            finalFeatures = new double[T][D];
            for (int t = 0; t < T; t++) 
            {
                cache.lnF[t] = layerNormForward(curX[t], gammaF, betaF, 1e-5);
                finalFeatures[t] = cache.lnF[t].y;
            }
        }

        // 4. Output Head (Wout + bout) -> Softmax -> Loss
        double totalLoss = 0.0;
        for (int t = 0; t < T; t++) 
        {
            double maxLogit = -1e9;
            for (int v = 0; v < V; v++) 
            {
                double score = bout[v];
                for (int d = 0; d < D; d++) score += finalFeatures[t][d] * Wout[d][v];
                cache.logits[t][v] = score;
                if (score > maxLogit) maxLogit = score;
            }

            double sumExp = 0.0;
            for (int v = 0; v < V; v++) 
            {
                cache.probs[t][v] = Math.exp(cache.logits[t][v] - maxLogit);
                sumExp += cache.probs[t][v];
            }
            for (int v = 0; v < V; v++) 
            {
                cache.probs[t][v] /= sumExp;
            }

            int target = targets[t];
            double pTarget = Math.max(cache.probs[t][target], 1e-15);
            cache.losses[t] = -Math.log(pTarget);
            totalLoss += cache.losses[t];
        }

        cache.avgLoss = totalLoss / T;
        return cache;
    }

    // =========================================================================
    // BACKWARD PASS
    // =========================================================================
    public Gradients backward(Cache cache) 
    {
        Gradients grads = new Gradients(V, T, D, nHeads, dk, H_mlp, L);
        double sqrtDk = Math.sqrt(dk);
        double invT = 1.0 / T;

        // 1. Loss gradients w.r.t logits: dLogit[t][v] = (p[v] - 1{v == target}) / T
        double[][] dFinalFeatures = new double[T][D];
        double[][] finalFeatures = (useLayerNorm) ? new double[T][D] : cache.blocks[L - 1].Z;
        if (useLayerNorm) 
        {
            for (int t = 0; t < T; t++) finalFeatures[t] = cache.lnF[t].y;
        }

        for (int t = 0; t < T; t++) 
        {
            int target = cache.targets[t];
            for (int v = 0; v < V; v++) 
            {
                double dLogit = (cache.probs[t][v] - (v == target ? 1.0 : 0.0)) * invT;
                grads.dbout[v] += dLogit;
                for (int d = 0; d < D; d++) 
                {
                    grads.dWout[d][v] += finalFeatures[t][d] * dLogit;
                    dFinalFeatures[t][d] += Wout[d][v] * dLogit;
                }
            }
        }

        // Backprop through Final LayerNorm
        double[][] dCurZ = dFinalFeatures;
        if (useLayerNorm) 
        {
            dCurZ = new double[T][D];
            for (int t = 0; t < T; t++) 
            {
                layerNormBackward(dFinalFeatures[t], cache.lnF[t], gammaF, dCurZ[t], grads.dGammaF, grads.dBetaF);
            }
        }

        // 2. Backprop through Transformer Blocks (reverse order)
        for (int l = L - 1; l >= 0; l--) 
        {
            Block b = blocks[l];
            BlockCache bc = cache.blocks[l];
            BlockGradients bg = grads.blockGrads[l];

            double[][] dZ = dCurZ;
            double[][] dY = new double[T][D];
            double[][] dNormY = new double[T][D];

            // FFN Residual: Z = Y + ffOut => dY += dZ, dFFOut = dZ
            for (int t = 0; t < T; t++) 
            {
                for (int d = 0; d < D; d++) 
                {
                    dY[t][d] += dZ[t][d];
                    double dff = dZ[t][d];
                    for (int h = 0; h < H_mlp; h++) 
                    {
                        bg.dW2[h][d] += bc.hDrop[t][h] * dff;
                        double dhDrop = b.W2[h][d] * dff;
                        double dhTanh = dhDrop * bc.dropMask[t][h];
                        double dhRaw = dhTanh * (1.0 - bc.hTanh[t][h] * bc.hTanh[t][h]);
                        bg.db1[h] += dhRaw;
                        for (int j = 0; j < D; j++) 
                        {
                            bg.dW1[j][h] += (useLayerNorm ? bc.ln2[t].y[j] : bc.Y[t][j]) * dhRaw;
                            dNormY[t][j] += b.W1[j][h] * dhRaw;
                        }
                    }
                    bg.db2[d] += dff;
                }
            }

            // Backprop through LayerNorm 2
            if (useLayerNorm) 
            {
                double[][] dYfromLN = new double[T][D];
                for (int t = 0; t < T; t++) 
                {
                    layerNormBackward(dNormY[t], bc.ln2[t], b.gamma2, dYfromLN[t], bg.dGamma2, bg.dBeta2);
                    for (int d = 0; d < D; d++) dY[t][d] += dYfromLN[t][d];
                }
            } 
            else 
            {
                for (int t = 0; t < T; t++) 
                    for (int d = 0; d < D; d++) dY[t][d] += dNormY[t][d];
            }

            // Attention Residual: Y = X + attOut => dX = dY, dAttOut = dY
            double[][] dX = new double[T][D];
            double[][] dHeadConcat = new double[T][D];
            for (int t = 0; t < T; t++) 
            {
                for (int d = 0; d < D; d++) 
                {
                    dX[t][d] += dY[t][d];
                    double datt = dY[t][d];
                    for (int j = 0; j < D; j++) 
                    {
                        bg.dWo[j][d] += bc.headConcat[t][j] * datt;
                        dHeadConcat[t][j] += b.Wo[j][d] * datt;
                    }
                }
            }

            // Multi-head attention backpropagation
            double[][] dNormX1 = new double[T][D];
            for (int h = 0; h < nHeads; h++) 
            {
                double[][] dQ = new double[T][dk];
                double[][] dK = new double[T][dk];
                double[][] dV = new double[T][dk];
                double[][] dScores = new double[T][T];

                for (int t = 0; t < T; t++) 
                {
                    // Gradient w.r.t attention blend: att_t = sum_i A[t][i] * V[i]
                    for (int k = 0; k < dk; k++) 
                    {
                        double dHead = dHeadConcat[t][h * dk + k];
                        for (int i = 0; i <= t; i++) 
                        {
                            dV[i][k] += bc.A[h][t][i] * dHead;
                            dScores[t][i] += bc.V[h][i][k] * dHead; // dA[t][i] accumulated into dScores temporarily
                        }
                    }

                    // Softmax backprop: dScore[t][i] = A[t][i] * (dA[t][i] - sum_j A[t][j]*dA[t][j])
                    double dotA = 0.0;
                    for (int i = 0; i <= t; i++) dotA += bc.A[h][t][i] * dScores[t][i];
                    for (int i = 0; i <= t; i++) 
                    {
                        double ds = bc.A[h][t][i] * (dScores[t][i] - dotA);
                        double dsScaled = ds / sqrtDk;
                        for (int k = 0; k < dk; k++) 
                        {
                            dQ[t][k] += bc.K[h][i][k] * dsScaled;
                            dK[i][k] += bc.Q[h][t][k] * dsScaled;
                        }
                    }
                }

                // Backprop Q, K, V to input and weights
                double[][] normX1 = (useLayerNorm) ? new double[T][D] : bc.X;
                if (useLayerNorm) 
                {
                    for (int t = 0; t < T; t++) normX1[t] = bc.ln1[t].y;
                }

                for (int t = 0; t < T; t++) 
                {
                    for (int k = 0; k < dk; k++) 
                    {
                        double dq = dQ[t][k];
                        double dkVal = dK[t][k];
                        double dv = dV[t][k];
                        for (int d = 0; d < D; d++) 
                        {
                            bg.dWq[h][d][k] += normX1[t][d] * dq;
                            bg.dWk[h][d][k] += normX1[t][d] * dkVal;
                            bg.dWv[h][d][k] += normX1[t][d] * dv;

                            dNormX1[t][d] += b.Wq[h][d][k] * dq + b.Wk[h][d][k] * dkVal + b.Wv[h][d][k] * dv;
                        }
                    }
                }
            }

            // Backprop through LayerNorm 1
            if (useLayerNorm) 
            {
                double[][] dXfromLN = new double[T][D];
                for (int t = 0; t < T; t++) 
                {
                    layerNormBackward(dNormX1[t], bc.ln1[t], b.gamma1, dXfromLN[t], bg.dGamma1, bg.dBeta1);
                    for (int d = 0; d < D; d++) dX[t][d] += dXfromLN[t][d];
                }
            } 
            else 
            {
                for (int t = 0; t < T; t++) 
                    for (int d = 0; d < D; d++) dX[t][d] += dNormX1[t][d];
            }

            dCurZ = dX;
        }

        // 3. Backprop through Embeddings: C[token] + P[t]
        for (int t = 0; t < T; t++) 
        {
            int tok = cache.tokens[t];
            for (int d = 0; d < D; d++) 
            {
                double dEmb = dCurZ[t][d];
                grads.dC[tok][d] += dEmb;
                grads.dP[t][d] += dEmb;
            }
        }

        return grads;
    }

    // =========================================================================
    // OPTIMIZERS: SGD WITH MOMENTUM & ADAM
    // =========================================================================
    public void stepAdam(Gradients g, double lr, double beta1, double beta2, double eps, double weightDecay) 
    {
        adam.t++;
        double biasCorr1 = 1.0 - Math.pow(beta1, adam.t);
        double biasCorr2 = 1.0 - Math.pow(beta2, adam.t);
        double stepSize = lr * Math.sqrt(biasCorr2) / biasCorr1;

        updateAdam2D(C, g.dC, adam.mC, adam.vC, beta1, beta2, stepSize, eps, weightDecay);
        updateAdam2D(P, g.dP, adam.mP, adam.vP, beta1, beta2, stepSize, eps, weightDecay);

        for (int l = 0; l < L; l++) 
        {
            Block b = blocks[l];
            BlockGradients bg = g.blockGrads[l];
            AdamState.BlockAdam ba = adam.blockAdam[l];

            updateAdam3D(b.Wq, bg.dWq, ba.mWq, ba.vWq, beta1, beta2, stepSize, eps, weightDecay);
            updateAdam3D(b.Wk, bg.dWk, ba.mWk, ba.vWk, beta1, beta2, stepSize, eps, weightDecay);
            updateAdam3D(b.Wv, bg.dWv, ba.mWv, ba.vWv, beta1, beta2, stepSize, eps, weightDecay);
            updateAdam2D(b.Wo, bg.dWo, ba.mWo, ba.vWo, beta1, beta2, stepSize, eps, weightDecay);

            updateAdam2D(b.W1, bg.dW1, ba.mW1, ba.vW1, beta1, beta2, stepSize, eps, weightDecay);
            updateAdam1D(b.b1, bg.db1, ba.mb1, ba.vb1, beta1, beta2, stepSize, eps, 0.0);
            updateAdam2D(b.W2, bg.dW2, ba.mW2, ba.vW2, beta1, beta2, stepSize, eps, weightDecay);
            updateAdam1D(b.b2, bg.db2, ba.mb2, ba.vb2, beta1, beta2, stepSize, eps, 0.0);

            if (useLayerNorm) 
            {
                updateAdam1D(b.gamma1, bg.dGamma1, ba.mGamma1, ba.vGamma1, beta1, beta2, stepSize, eps, 0.0);
                updateAdam1D(b.beta1, bg.dBeta1, ba.mBeta1, ba.vBeta1, beta1, beta2, stepSize, eps, 0.0);
                updateAdam1D(b.gamma2, bg.dGamma2, ba.mGamma2, ba.vGamma2, beta1, beta2, stepSize, eps, 0.0);
                updateAdam1D(b.beta2, bg.dBeta2, ba.mBeta2, ba.vBeta2, beta1, beta2, stepSize, eps, 0.0);
            }
        }

        if (useLayerNorm) 
        {
            updateAdam1D(gammaF, g.dGammaF, adam.mGammaF, adam.vGammaF, beta1, beta2, stepSize, eps, 0.0);
            updateAdam1D(betaF, g.dBetaF, adam.mBetaF, adam.vBetaF, beta1, beta2, stepSize, eps, 0.0);
        }

        updateAdam2D(Wout, g.dWout, adam.mWout, adam.vWout, beta1, beta2, stepSize, eps, weightDecay);
        updateAdam1D(bout, g.dbout, adam.mbout, adam.vbout, beta1, beta2, stepSize, eps, 0.0);
    }

    private static void updateAdam1D(double[] param, double[] grad, double[] m, double[] v, double b1, double b2, double step, double eps, double wd) 
    {
        for (int i = 0; i < param.length; i++) 
        {
            double g = grad[i] + (wd > 0.0 ? wd * param[i] : 0.0);
            m[i] = b1 * m[i] + (1.0 - b1) * g;
            v[i] = b2 * v[i] + (1.0 - b2) * g * g;
            param[i] -= step * (m[i] / (Math.sqrt(v[i]) + eps));
        }
    }

    private static void updateAdam2D(double[][] param, double[][] grad, double[][] m, double[][] v, double b1, double b2, double step, double eps, double wd) 
    {
        for (int i = 0; i < param.length; i++) updateAdam1D(param[i], grad[i], m[i], v[i], b1, b2, step, eps, wd);
    }

    private static void updateAdam3D(double[][][] param, double[][][] grad, double[][][] m, double[][][] v, double b1, double b2, double step, double eps, double wd) 
    {
        for (int i = 0; i < param.length; i++) updateAdam2D(param[i], grad[i], m[i], v[i], b1, b2, step, eps, wd);
    }

    public void stepSGD(Gradients g, double lr, double weightDecay) 
    {
        updateSGD2D(C, g.dC, lr, weightDecay);
        updateSGD2D(P, g.dP, lr, weightDecay);
        for (int l = 0; l < L; l++) 
        {
            Block b = blocks[l];
            BlockGradients bg = g.blockGrads[l];
            updateSGD3D(b.Wq, bg.dWq, lr, weightDecay);
            updateSGD3D(b.Wk, bg.dWk, lr, weightDecay);
            updateSGD3D(b.Wv, bg.dWv, lr, weightDecay);
            updateSGD2D(b.Wo, bg.dWo, lr, weightDecay);

            updateSGD2D(b.W1, bg.dW1, lr, weightDecay);
            updateSGD1D(b.b1, bg.db1, lr, 0.0);
            updateSGD2D(b.W2, bg.dW2, lr, weightDecay);
            updateSGD1D(b.b2, bg.db2, lr, 0.0);

            if (useLayerNorm) 
            {
                updateSGD1D(b.gamma1, bg.dGamma1, lr, 0.0);
                updateSGD1D(b.beta1, bg.dBeta1, lr, 0.0);
                updateSGD1D(b.gamma2, bg.dGamma2, lr, 0.0);
                updateSGD1D(b.beta2, bg.dBeta2, lr, 0.0);
            }
        }
        if (useLayerNorm) 
        {
            updateSGD1D(gammaF, g.dGammaF, lr, 0.0);
            updateSGD1D(betaF, g.dBetaF, lr, 0.0);
        }
        updateSGD2D(Wout, g.dWout, lr, weightDecay);
        updateSGD1D(bout, g.dbout, lr, 0.0);
    }

    private static void updateSGD1D(double[] param, double[] grad, double lr, double wd) 
    {
        for (int i = 0; i < param.length; i++) param[i] -= lr * (grad[i] + (wd > 0.0 ? wd * param[i] : 0.0));
    }

    private static void updateSGD2D(double[][] param, double[][] grad, double lr, double wd) 
    {
        for (int i = 0; i < param.length; i++) updateSGD1D(param[i], grad[i], lr, wd);
    }

    private static void updateSGD3D(double[][][] param, double[][][] grad, double lr, double wd) 
    {
        for (int i = 0; i < param.length; i++) updateSGD2D(param[i], grad[i], lr, wd);
    }

    // =========================================================================
    // TEXT GENERATION WITH TEMPERATURE
    // =========================================================================
    public String generate(Vocab vocab, String prompt, int genLength, double temperature) 
    {
        int[] promptTokens = vocab.encodeString(prompt);
        int[] window = new int[T];
        int start = Math.max(0, promptTokens.length - T);
        int offset = Math.max(0, T - promptTokens.length);

        for (int i = 0; i < T; i++) 
        {
            if (i < offset) window[i] = vocab.encode(' ');
            else window[i] = promptTokens[start + (i - offset)];
        }

        StringBuilder sb = new StringBuilder(prompt);
        int[] dummyTargets = new int[T];

        for (int step = 0; step < genLength; step++) 
        {
            Cache cache = forward(window, dummyTargets, false, 0.0);
            double[] logits = cache.logits[T - 1].clone();

            // Temperature scaling
            double temp = Math.max(temperature, 1e-4);
            double maxLogit = -1e9;
            for (int v = 0; v < V; v++) 
            {
                logits[v] /= temp;
                if (logits[v] > maxLogit) maxLogit = logits[v];
            }

            double sumExp = 0.0;
            double[] probs = new double[V];
            for (int v = 0; v < V; v++) 
            {
                probs[v] = Math.exp(logits[v] - maxLogit);
                sumExp += probs[v];
            }
            for (int v = 0; v < V; v++) probs[v] /= sumExp;

            // Multinomial sample
            double r = ThreadLocalRandom.current().nextDouble();
            double cdf = 0.0;
            int nextTok = V - 1;
            for (int v = 0; v < V; v++) 
            {
                cdf += probs[v];
                if (r <= cdf) 
                {
                    nextTok = v;
                    break;
                }
            }

            char nextChar = vocab.decode(nextTok);
            sb.append(nextChar);

            // Shift window
            for (int i = 0; i < T - 1; i++) window[i] = window[i + 1];
            window[T - 1] = nextTok;
        }

        return sb.toString();
    }

    // =========================================================================
    // SERIALIZATION & DESERIALIZATION WITH OPTIMIZER STATE
    // =========================================================================
    public void saveCheckpoint(String filePath) throws IOException 
    {
        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filePath)))) 
        {
            dos.writeInt(0xEB8A1101); // Magic header for E-Brain Stage 11
            dos.writeInt(V);
            dos.writeInt(T);
            dos.writeInt(D);
            dos.writeInt(nHeads);
            dos.writeInt(dk);
            dos.writeInt(H_mlp);
            dos.writeInt(L);
            dos.writeBoolean(useLayerNorm);
            dos.writeLong(adam != null ? adam.t : 0);

            // Write C & P
            write2D(dos, C);
            write2D(dos, P);

            // Write Blocks
            for (int l = 0; l < L; l++) 
            {
                Block b = blocks[l];
                write3D(dos, b.Wq);
                write3D(dos, b.Wk);
                write3D(dos, b.Wv);
                write2D(dos, b.Wo);
                write2D(dos, b.W1);
                write1D(dos, b.b1);
                write2D(dos, b.W2);
                write1D(dos, b.b2);
                write1D(dos, b.gamma1);
                write1D(dos, b.beta1);
                write1D(dos, b.gamma2);
                write1D(dos, b.beta2);
            }

            // Write Head & Final LN
            write1D(dos, gammaF);
            write1D(dos, betaF);
            write2D(dos, Wout);
            write1D(dos, bout);

            // Write Adam state if present
            boolean hasAdam = (adam != null && adam.t > 0);
            dos.writeBoolean(hasAdam);
            if (hasAdam) 
            {
                write2D(dos, adam.mC); write2D(dos, adam.vC);
                write2D(dos, adam.mP); write2D(dos, adam.vP);
                write2D(dos, adam.mWout); write2D(dos, adam.vWout);
                write1D(dos, adam.mbout); write1D(dos, adam.vbout);
                write1D(dos, adam.mGammaF); write1D(dos, adam.vGammaF);
                write1D(dos, adam.mBetaF); write1D(dos, adam.vBetaF);

                for (int l = 0; l < L; l++) 
                {
                    AdamState.BlockAdam ba = adam.blockAdam[l];
                    write3D(dos, ba.mWq); write3D(dos, ba.vWq);
                    write3D(dos, ba.mWk); write3D(dos, ba.vWk);
                    write3D(dos, ba.mWv); write3D(dos, ba.vWv);
                    write2D(dos, ba.mWo); write2D(dos, ba.vWo);
                    write2D(dos, ba.mW1); write2D(dos, ba.vW1);
                    write1D(dos, ba.mb1); write1D(dos, ba.vb1);
                    write2D(dos, ba.mW2); write2D(dos, ba.vW2);
                    write1D(dos, ba.mb2); write1D(dos, ba.vb2);
                    write1D(dos, ba.mGamma1); write1D(dos, ba.vGamma1);
                    write1D(dos, ba.mBeta1); write1D(dos, ba.vBeta1);
                    write1D(dos, ba.mGamma2); write1D(dos, ba.vGamma2);
                    write1D(dos, ba.mBeta2); write1D(dos, ba.vBeta2);
                }
            }
        }
    }

    public static Transformer loadCheckpoint(String filePath) throws IOException 
    {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) 
        {
            int magic = dis.readInt();
            if (magic != 0xEB8A1101) 
            {
                throw new IOException("Invalid checkpoint format magic: " + Integer.toHexString(magic));
            }

            Transformer model = new Transformer();
            model.V = dis.readInt();
            model.T = dis.readInt();
            model.D = dis.readInt();
            model.nHeads = dis.readInt();
            model.dk = dis.readInt();
            model.H_mlp = dis.readInt();
            model.L = dis.readInt();
            model.useLayerNorm = dis.readBoolean();
            long adamT = dis.readLong();

            model.C = read2D(dis, model.V, model.D);
            model.P = read2D(dis, model.T, model.D);

            model.blocks = new Block[model.L];
            for (int l = 0; l < model.L; l++) 
            {
                Block b = new Block();
                b.Wq = read3D(dis, model.nHeads, model.D, model.dk);
                b.Wk = read3D(dis, model.nHeads, model.D, model.dk);
                b.Wv = read3D(dis, model.nHeads, model.D, model.dk);
                b.Wo = read2D(dis, model.D, model.D);
                b.W1 = read2D(dis, model.D, model.H_mlp);
                b.b1 = read1D(dis, model.H_mlp);
                b.W2 = read2D(dis, model.H_mlp, model.D);
                b.b2 = read1D(dis, model.D);
                b.gamma1 = read1D(dis, model.D);
                b.beta1 = read1D(dis, model.D);
                b.gamma2 = read1D(dis, model.D);
                b.beta2 = read1D(dis, model.D);
                model.blocks[l] = b;
            }

            model.gammaF = read1D(dis, model.D);
            model.betaF = read1D(dis, model.D);
            model.Wout = read2D(dis, model.D, model.V);
            model.bout = read1D(dis, model.V);

            model.initAdam();
            model.adam.t = adamT;

            boolean hasAdam = dis.readBoolean();
            if (hasAdam) 
            {
                model.adam.mC = read2D(dis, model.V, model.D);
                model.adam.vC = read2D(dis, model.V, model.D);
                model.adam.mP = read2D(dis, model.T, model.D);
                model.adam.vP = read2D(dis, model.T, model.D);
                model.adam.mWout = read2D(dis, model.D, model.V);
                model.adam.vWout = read2D(dis, model.D, model.V);
                model.adam.mbout = read1D(dis, model.V);
                model.adam.vbout = read1D(dis, model.V);
                model.adam.mGammaF = read1D(dis, model.D);
                model.adam.vGammaF = read1D(dis, model.D);
                model.adam.mBetaF = read1D(dis, model.D);
                model.adam.vBetaF = read1D(dis, model.D);

                for (int l = 0; l < model.L; l++) 
                {
                    AdamState.BlockAdam ba = model.adam.blockAdam[l];
                    ba.mWq = read3D(dis, model.nHeads, model.D, model.dk);
                    ba.vWq = read3D(dis, model.nHeads, model.D, model.dk);
                    ba.mWk = read3D(dis, model.nHeads, model.D, model.dk);
                    ba.vWk = read3D(dis, model.nHeads, model.D, model.dk);
                    ba.mWv = read3D(dis, model.nHeads, model.D, model.dk);
                    ba.vWv = read3D(dis, model.nHeads, model.D, model.dk);
                    ba.mWo = read2D(dis, model.D, model.D);
                    ba.vWo = read2D(dis, model.D, model.D);
                    ba.mW1 = read2D(dis, model.D, model.H_mlp);
                    ba.vW1 = read2D(dis, model.D, model.H_mlp);
                    ba.mb1 = read1D(dis, model.H_mlp);
                    ba.vb1 = read1D(dis, model.H_mlp);
                    ba.mW2 = read2D(dis, model.H_mlp, model.D);
                    ba.vW2 = read2D(dis, model.H_mlp, model.D);
                    ba.mb2 = read1D(dis, model.D);
                    ba.vb2 = read1D(dis, model.D);
                    ba.mGamma1 = read1D(dis, model.D);
                    ba.vGamma1 = read1D(dis, model.D);
                    ba.mBeta1 = read1D(dis, model.D);
                    ba.vBeta1 = read1D(dis, model.D);
                    ba.mGamma2 = read1D(dis, model.D);
                    ba.vGamma2 = read1D(dis, model.D);
                    ba.mBeta2 = read1D(dis, model.D);
                    ba.vBeta2 = read1D(dis, model.D);
                }
            }

            model.accumGrads = new Gradients(model.V, model.T, model.D, model.nHeads, model.dk, model.H_mlp, model.L);
            return model;
        }
    }

    // Binary I/O helpers
    private static void write1D(DataOutputStream dos, double[] a) throws IOException 
    {
        for (double v : a) dos.writeDouble(v);
    }
    private static void write2D(DataOutputStream dos, double[][] a) throws IOException 
    {
        for (double[] r : a) write1D(dos, r);
    }
    private static void write3D(DataOutputStream dos, double[][][] a) throws IOException 
    {
        for (double[][] m : a) write2D(dos, m);
    }
    private static double[] read1D(DataInputStream dis, int n) throws IOException 
    {
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = dis.readDouble();
        return a;
    }
    private static double[][] read2D(DataInputStream dis, int r, int c) throws IOException 
    {
        double[][] a = new double[r][c];
        for (int i = 0; i < r; i++) a[i] = read1D(dis, c);
        return a;
    }
    private static double[][][] read3D(DataInputStream dis, int d1, int d2, int d3) throws IOException 
    {
        double[][][] a = new double[d1][d2][d3];
        for (int i = 0; i < d1; i++) a[i] = read2D(dis, d2, d3);
        return a;
    }

    // Basic vector/matrix utilities
    public static double[][] randomMatrix(int rows, int cols, double scale, Random rng) 
    {
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++) 
            for (int j = 0; j < cols; j++) 
                m[i][j] = (rng.nextDouble() * 2.0 - 1.0) * scale;
        return m;
    }

    public static double[] onesVector(int len) 
    {
        double[] v = new double[len];
        Arrays.fill(v, 1.0);
        return v;
    }

    public static void zero1D(double[] a) { Arrays.fill(a, 0.0); }
    public static void zero2D(double[][] a) { for (double[] r : a) zero1D(r); }
    public static void zero3D(double[][][] a) { for (double[][] m : a) zero2D(m); }

    public static void add1D(double[] a, double[] b) { for (int i = 0; i < a.length; i++) a[i] += b[i]; }
    public static void add2D(double[][] a, double[][] b) { for (int i = 0; i < a.length; i++) add1D(a[i], b[i]); }
    public static void add3D(double[][][] a, double[][][] b) { for (int i = 0; i < a.length; i++) add2D(a[i], b[i]); }

    public static void scale1D(double[] a, double s) { for (int i = 0; i < a.length; i++) a[i] *= s; }
    public static void scale2D(double[][] a, double s) { for (int i = 0; i < a.length; i++) scale1D(a[i], s); }
    public static void scale3D(double[][][] a, double s) { for (int i = 0; i < a.length; i++) scale2D(a[i], s); }
}
