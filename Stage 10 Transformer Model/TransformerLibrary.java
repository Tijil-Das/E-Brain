import java.util.concurrent.ThreadLocalRandom;

public class TransformerLibrary 
{
    static char[] ch_present;

    public static void fill_the_characters_present_in_text(String data)
    {
        char[] temp = new char[256];
        int c = 0;
        for(int i = 0; i < data.length(); i++)
        {
            char ch = data.charAt(i);
            boolean found = false;
            for(int j = 0; j < c; j++)
            {
                if(temp[j] == ch)
                {
                    found = true;
                    break;
                }
            }
            if(!found)
            {
                temp[c] = ch;
                c++;
            }
        }

        ch_present = new char[c];
        for(int i = 0; i < c; i++)
        {
            ch_present[i] = temp[i];
        }
    }

    public static int find_ch_number(char ch, char[] array)
    {
        for(int i = 0; i < array.length; i++)
        {
            if(array[i] == ch)
                return i;
        }
        return -1;
    }

    public static Character find_ch_at_a_number(int n, char[] array)
    {
        if(n < 0 || n >= array.length)
            return null;
        return array[n];
    }

    public static double[][] make_random_matrix(int rows, int cols, double scale)
    {
        double[][] mat = new double[rows][cols];
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] = ThreadLocalRandom.current().nextDouble(-scale, scale);
        return mat;
    }

    public static double[] make_zeros_vector(int len)
    {
        return new double[len];
    }

    public static double relative_error(double a, double n)
    {
        return Math.abs(a - n) / (Math.abs(a) + Math.abs(n) + 1e-12);
    }

    // =========================================================================
    // TRANSFORMER BLOCK & MODEL ARCHITECTURE
    // =========================================================================
    public static class BlockWeights
    {
        // Multi-head attention (H heads)
        public double[][][] Wq; // [H][D][dk]
        public double[][][] Wk; // [H][D][dk]
        public double[][][] Wv; // [H][D][dk]

        // Feed-Forward: D -> H_mlp -> D
        public double[][] W1;   // [D][H_mlp]
        public double[] b1;     // [H_mlp]
        public double[][] W2;   // [H_mlp][D]
        public double[] b2;     // [D]

        public static BlockWeights init(int D, int n_heads, int H_mlp, double scale)
        {
            BlockWeights bw = new BlockWeights();
            int dk = D / n_heads;
            bw.Wq = new double[n_heads][D][dk];
            bw.Wk = new double[n_heads][D][dk];
            bw.Wv = new double[n_heads][D][dk];
            for(int h = 0; h < n_heads; h++)
            {
                bw.Wq[h] = make_random_matrix(D, dk, scale);
                bw.Wk[h] = make_random_matrix(D, dk, scale);
                bw.Wv[h] = make_random_matrix(D, dk, scale);
            }
            bw.W1 = make_random_matrix(D, H_mlp, scale);
            bw.b1 = make_zeros_vector(H_mlp);
            bw.W2 = make_random_matrix(H_mlp, D, scale);
            bw.b2 = make_zeros_vector(D);
            return bw;
        }

        public BlockWeights copy()
        {
            BlockWeights cp = new BlockWeights();
            int n_heads = Wq.length;
            int D = Wq[0].length;
            int dk = Wq[0][0].length;
            int H_mlp = b1.length;

            cp.Wq = new double[n_heads][D][dk];
            cp.Wk = new double[n_heads][D][dk];
            cp.Wv = new double[n_heads][D][dk];
            for(int h = 0; h < n_heads; h++)
            {
                for(int r = 0; r < D; r++)
                {
                    cp.Wq[h][r] = Wq[h][r].clone();
                    cp.Wk[h][r] = Wk[h][r].clone();
                    cp.Wv[h][r] = Wv[h][r].clone();
                }
            }

            cp.W1 = new double[D][H_mlp];
            for(int r = 0; r < D; r++) cp.W1[r] = W1[r].clone();
            cp.b1 = b1.clone();

            cp.W2 = new double[H_mlp][D];
            for(int r = 0; r < H_mlp; r++) cp.W2[r] = W2[r].clone();
            cp.b2 = b2.clone();

            return cp;
        }
    }

    public static class TransformerModel
    {
        public double[][] C;       // [V][D]
        public double[][] P;       // [T][D]
        public BlockWeights[] blocks; // L blocks
        public double[][] Wout;    // [D][V]
        public double[] bout;      // [V]

        public int V, T, D, n_heads, H_mlp, L;

        public static TransformerModel init(int V, int T, int D, int n_heads, int H_mlp, int L, double scale)
        {
            TransformerModel tm = new TransformerModel();
            tm.V = V; tm.T = T; tm.D = D; tm.n_heads = n_heads; tm.H_mlp = H_mlp; tm.L = L;
            tm.C = make_random_matrix(V, D, scale);
            tm.P = make_random_matrix(T, D, scale);

            tm.blocks = new BlockWeights[L];
            for(int l = 0; l < L; l++)
                tm.blocks[l] = BlockWeights.init(D, n_heads, H_mlp, scale);

            tm.Wout = make_random_matrix(D, V, scale);
            tm.bout = make_zeros_vector(V);
            return tm;
        }

        public TransformerModel copy()
        {
            TransformerModel cp = new TransformerModel();
            cp.V = V; cp.T = T; cp.D = D; cp.n_heads = n_heads; cp.H_mlp = H_mlp; cp.L = L;
            cp.C = new double[V][D];
            for(int i = 0; i < V; i++) cp.C[i] = C[i].clone();
            cp.P = new double[T][D];
            for(int i = 0; i < T; i++) cp.P[i] = P[i].clone();

            cp.blocks = new BlockWeights[L];
            for(int l = 0; l < L; l++) cp.blocks[l] = blocks[l].copy();

            cp.Wout = new double[D][V];
            for(int i = 0; i < D; i++) cp.Wout[i] = Wout[i].clone();
            cp.bout = bout.clone();
            return cp;
        }
    }

    // =========================================================================
    // FORWARD & BACKWARD DATA STRUCTURES
    // =========================================================================
    public static class BlockActivation
    {
        public double[][] X;          // [T][D] input to block
        public double[][][] Q;        // [H][T][dk]
        public double[][][] K;        // [H][T][dk]
        public double[][][] V;        // [H][T][dk]
        public double[][][] scores;   // [H][T][T]
        public double[][][] A;        // [H][T][T] softmax attention weights
        public double[][] att_out;    // [T][D]
        public double[][] Y;          // [T][D] after attention residual (X + att_out)
        public double[][] h_mlp;      // [T][H_mlp] hidden tanh
        public double[][] ff_out;     // [T][D]
        public double[][] Z;          // [T][D] after ff residual (Y + ff_out)

        public BlockActivation(int T, int D, int n_heads, int H_mlp)
        {
            int dk = D / n_heads;
            X = new double[T][D];
            Q = new double[n_heads][T][dk];
            K = new double[n_heads][T][dk];
            V = new double[n_heads][T][dk];
            scores = new double[n_heads][T][T];
            A = new double[n_heads][T][T];
            att_out = new double[T][D];
            Y = new double[T][D];
            h_mlp = new double[T][H_mlp];
            ff_out = new double[T][D];
            Z = new double[T][D];
        }
    }

    public static class ForwardCache
    {
        public BlockActivation[] block_acts;
        public double[][] final_scores; // [T][V]
        public double[][] probs;        // [T][V]
        public double[] pos_losses;     // [T]
        public double total_loss;       // average over T positions

        public ForwardCache(int L, int T, int D, int n_heads, int H_mlp, int V)
        {
            block_acts = new BlockActivation[L];
            for(int l = 0; l < L; l++)
                block_acts[l] = new BlockActivation(T, D, n_heads, H_mlp);
            final_scores = new double[T][V];
            probs = new double[T][V];
            pos_losses = new double[T];
        }
    }

    public static ForwardCache forward(int[] window_chars, int[] target_chars,
                                       TransformerModel model)
    {
        int T = model.T;
        int D = model.D;
        int n_heads = model.n_heads;
        int dk = D / n_heads;
        int H_mlp = model.H_mlp;
        int V = model.V;
        int L = model.L;
        double scale = Math.sqrt(dk);

        ForwardCache fc = new ForwardCache(L, T, D, n_heads, H_mlp, V);

        // Initial embedding: X0[t] = C[char_t] + P[t]
        double[][] cur_X = new double[T][D];
        for(int t = 0; t < T; t++)
        {
            int c = window_chars[t];
            for(int m = 0; m < D; m++)
                cur_X[t][m] = model.C[c][m] + model.P[t][m];
        }

        for(int l = 0; l < L; l++)
        {
            BlockActivation ba = fc.block_acts[l];
            BlockWeights bw = model.blocks[l];

            // Save input X to block
            for(int t = 0; t < T; t++)
                ba.X[t] = cur_X[t].clone();

            // Compute Q, K, V for all positions and heads
            for(int h = 0; h < n_heads; h++)
            {
                for(int t = 0; t < T; t++)
                {
                    for(int col = 0; col < dk; col++)
                    {
                        double sum_q = 0.0, sum_k = 0.0, sum_v = 0.0;
                        for(int row = 0; row < D; row++)
                        {
                            double x_val = ba.X[t][row];
                            sum_q += x_val * bw.Wq[h][row][col];
                            sum_k += x_val * bw.Wk[h][row][col];
                            sum_v += x_val * bw.Wv[h][row][col];
                        }
                        ba.Q[h][t][col] = sum_q;
                        ba.K[h][t][col] = sum_k;
                        ba.V[h][t][col] = sum_v;
                    }
                }

                // Causal Attention Scores & Softmax
                for(int t = 0; t < T; t++)
                {
                    double max_s = -Double.MAX_VALUE;
                    for(int i = 0; i < T; i++)
                    {
                        if(i > t)
                        {
                            ba.scores[h][t][i] = -1e9; // Causal mask: future positions invisible
                        }
                        else
                        {
                            double dot = 0.0;
                            for(int m = 0; m < dk; m++)
                                dot += ba.Q[h][t][m] * ba.K[h][i][m];
                            ba.scores[h][t][i] = dot / scale;
                            if(ba.scores[h][t][i] > max_s)
                                max_s = ba.scores[h][t][i];
                        }
                    }

                    double sum_exp = 0.0;
                    for(int i = 0; i <= t; i++)
                    {
                        ba.A[h][t][i] = Math.exp(ba.scores[h][t][i] - max_s);
                        sum_exp += ba.A[h][t][i];
                    }
                    for(int i = 0; i <= t; i++)
                        ba.A[h][t][i] /= sum_exp;
                    for(int i = t + 1; i < T; i++)
                        ba.A[h][t][i] = 0.0;

                    // Blend head output
                    int offset = h * dk;
                    for(int m = 0; m < dk; m++)
                    {
                        double blend = 0.0;
                        for(int i = 0; i <= t; i++)
                            blend += ba.A[h][t][i] * ba.V[h][i][m];
                        ba.att_out[t][offset + m] = blend;
                    }
                }
            }

            // Residual 1: Y = X + att_out
            // Feed-Forward: h_mlp = tanh(Y * W1 + b1), ff_out = h_mlp * W2 + b2
            // Residual 2: Z = Y + ff_out
            for(int t = 0; t < T; t++)
            {
                for(int m = 0; m < D; m++)
                    ba.Y[t][m] = ba.X[t][m] + ba.att_out[t][m];

                for(int col = 0; col < H_mlp; col++)
                {
                    double sum = 0.0;
                    for(int row = 0; row < D; row++)
                        sum += ba.Y[t][row] * bw.W1[row][col];
                    ba.h_mlp[t][col] = Math.tanh(sum + bw.b1[col]);
                }

                for(int col = 0; col < D; col++)
                {
                    double sum = 0.0;
                    for(int row = 0; row < H_mlp; row++)
                        sum += ba.h_mlp[t][row] * bw.W2[row][col];
                    ba.ff_out[t][col] = sum + bw.b2[col];
                    ba.Z[t][col] = ba.Y[t][col] + ba.ff_out[t][col];
                }
            }

            cur_X = ba.Z;
        }

        // Final linear layer & Cross-Entropy loss at every position
        double sum_loss = 0.0;
        for(int t = 0; t < T; t++)
        {
            double max_out = -Double.MAX_VALUE;
            for(int col = 0; col < V; col++)
            {
                double sum = 0.0;
                for(int row = 0; row < D; row++)
                    sum += cur_X[t][row] * model.Wout[row][col];
                fc.final_scores[t][col] = sum + model.bout[col];
                if(fc.final_scores[t][col] > max_out)
                    max_out = fc.final_scores[t][col];
            }

            double sum_exp_out = 0.0;
            for(int j = 0; j < V; j++)
            {
                fc.probs[t][j] = Math.exp(fc.final_scores[t][j] - max_out);
                sum_exp_out += fc.probs[t][j];
            }
            for(int j = 0; j < V; j++)
                fc.probs[t][j] /= sum_exp_out;

            if(target_chars != null && target_chars[t] >= 0)
            {
                fc.pos_losses[t] = -Math.log(Math.max(fc.probs[t][target_chars[t]], 1e-15));
                sum_loss += fc.pos_losses[t];
            }
        }

        fc.total_loss = sum_loss / T;
        return fc;
    }

    // =========================================================================
    // BACKWARD PASS
    // =========================================================================
    public static void backward(ForwardCache fc, int[] window_chars, int[] target_chars,
                                TransformerModel model, double lr)
    {
        int T = model.T;
        int D = model.D;
        int n_heads = model.n_heads;
        int dk = D / n_heads;
        int H_mlp = model.H_mlp;
        int V = model.V;
        int L = model.L;
        double scale = Math.sqrt(dk);

        // 1. Output Layer Gradients
        double[][] g = new double[T][V];
        double[][] dZ = new double[T][D];

        for(int t = 0; t < T; t++)
        {
            int target = target_chars[t];
            for(int j = 0; j < V; j++)
                g[t][j] = (fc.probs[t][j] - ((j == target) ? 1.0 : 0.0)) / T;

            for(int row = 0; row < D; row++)
            {
                double sum_d = 0.0;
                for(int col = 0; col < V; col++)
                    sum_d += g[t][col] * model.Wout[row][col];
                dZ[t][row] = sum_d;
            }
        }

        // Update Wout and bout
        for(int t = 0; t < T; t++)
        {
            double[] z_t = fc.block_acts[L - 1].Z[t];
            for(int col = 0; col < V; col++)
            {
                model.bout[col] -= lr * g[t][col];
                for(int row = 0; row < D; row++)
                    model.Wout[row][col] -= lr * z_t[row] * g[t][col];
            }
        }

        // 2. Backprop through blocks in reverse order
        for(int l = L - 1; l >= 0; l--)
        {
            BlockActivation ba = fc.block_acts[l];
            BlockWeights bw = model.blocks[l];

            // Feed-Forward backward: Z = Y + ff_out
            double[][] dY = new double[T][D];
            double[][] dz_mlp = new double[T][H_mlp];

            for(int t = 0; t < T; t++)
            {
                for(int m = 0; m < D; m++)
                    dY[t][m] = dZ[t][m]; // residual branch

                for(int row = 0; row < H_mlp; row++)
                {
                    double sum_h = 0.0;
                    for(int col = 0; col < D; col++)
                        sum_h += dZ[t][col] * bw.W2[row][col];
                    double h_val = ba.h_mlp[t][row];
                    dz_mlp[t][row] = sum_h * (1.0 - h_val * h_val);
                }

                for(int row = 0; row < D; row++)
                {
                    double sum_y = 0.0;
                    for(int col = 0; col < H_mlp; col++)
                        sum_y += dz_mlp[t][col] * bw.W1[row][col];
                    dY[t][row] += sum_y;
                }
            }

            // Attention backward: Y = X + att_out
            double[][] dX = new double[T][D];
            for(int t = 0; t < T; t++)
                for(int m = 0; m < D; m++)
                    dX[t][m] = dY[t][m]; // residual branch

            double[][][] dQ = new double[n_heads][T][dk];
            double[][][] dK = new double[n_heads][T][dk];
            double[][][] dV = new double[n_heads][T][dk];

            for(int h = 0; h < n_heads; h++)
            {
                int offset = h * dk;

                // Through attention blend: att_out[t] = sum_{i<=t} A[t][i] * V[i]
                for(int t = 0; t < T; t++)
                {
                    double[] d_out_head = new double[dk];
                    for(int m = 0; m < dk; m++)
                        d_out_head[m] = dY[t][offset + m];

                    double[] dA_t = new double[T];
                    double sum_A_dA = 0.0;
                    for(int i = 0; i <= t; i++)
                    {
                        double dot = 0.0;
                        for(int m = 0; m < dk; m++)
                        {
                            dV[h][i][m] += ba.A[h][t][i] * d_out_head[m];
                            dot += d_out_head[m] * ba.V[h][i][m];
                        }
                        dA_t[i] = dot;
                        sum_A_dA += ba.A[h][t][i] * dot;
                    }

                    // Softmax backward for row t
                    for(int i = 0; i <= t; i++)
                    {
                        double ds = ba.A[h][t][i] * (dA_t[i] - sum_A_dA);
                        double ds_scaled = ds / scale;
                        for(int m = 0; m < dk; m++)
                        {
                            dQ[h][t][m] += ds_scaled * ba.K[h][i][m];
                            dK[h][i][m] += ds_scaled * ba.Q[h][t][m];
                        }
                    }
                }

                // Accumulate dX from head h
                for(int t = 0; t < T; t++)
                {
                    for(int row = 0; row < D; row++)
                    {
                        double sum_q = 0.0, sum_k = 0.0, sum_v = 0.0;
                        for(int col = 0; col < dk; col++)
                        {
                            sum_q += dQ[h][t][col] * bw.Wq[h][row][col];
                            sum_k += dK[h][t][col] * bw.Wk[h][row][col];
                            sum_v += dV[h][t][col] * bw.Wv[h][row][col];
                        }
                        dX[t][row] += sum_q + sum_k + sum_v;
                    }
                }
            }

            // Update Feed-Forward weights W1, b1, W2, b2
            for(int t = 0; t < T; t++)
            {
                for(int col = 0; col < D; col++)
                {
                    bw.b2[col] -= lr * dZ[t][col];
                    for(int row = 0; row < H_mlp; row++)
                        bw.W2[row][col] -= lr * ba.h_mlp[t][row] * dZ[t][col];
                }

                for(int col = 0; col < H_mlp; col++)
                {
                    bw.b1[col] -= lr * dz_mlp[t][col];
                    for(int row = 0; row < D; row++)
                        bw.W1[row][col] -= lr * ba.Y[t][row] * dz_mlp[t][col];
                }
            }

            // Update Attention weights Wq, Wk, Wv for all heads
            for(int h = 0; h < n_heads; h++)
            {
                for(int row = 0; row < D; row++)
                {
                    for(int col = 0; col < dk; col++)
                    {
                        double sum_wq = 0.0, sum_wk = 0.0, sum_wv = 0.0;
                        for(int t = 0; t < T; t++)
                        {
                            double x_val = ba.X[t][row];
                            sum_wq += x_val * dQ[h][t][col];
                            sum_wk += x_val * dK[h][t][col];
                            sum_wv += x_val * dV[h][t][col];
                        }
                        bw.Wq[h][row][col] -= lr * sum_wq;
                        bw.Wk[h][row][col] -= lr * sum_wk;
                        bw.Wv[h][row][col] -= lr * sum_wv;
                    }
                }
            }

            // For the next block down, dZ is dX of current block
            dZ = dX;
        }

        // Update token embeddings C and position embeddings P
        for(int t = 0; t < T; t++)
        {
            int c = window_chars[t];
            for(int m = 0; m < D; m++)
            {
                double grad = dZ[t][m];
                model.C[c][m] -= lr * grad;
                model.P[t][m] -= lr * grad;
            }
        }
    }

    // =========================================================================
    // CAUSALITY TEST
    // =========================================================================
    public static boolean run_causality_test(TransformerModel model, char[] vocab)
    {
        int T = model.T;
        int[] window1 = new int[T];
        int[] window2 = new int[T];

        for(int i = 0; i < T; i++)
        {
            window1[i] = i % vocab.length;
            window2[i] = i % vocab.length;
        }

        // Modify character at position 10 in window2
        int pos_modified = 10;
        window2[pos_modified] = (window1[pos_modified] + 7) % vocab.length;

        ForwardCache fc1 = forward(window1, null, model);
        ForwardCache fc2 = forward(window2, null, model);

        double max_diff_prior = 0.0;
        for(int t = 0; t < pos_modified; t++)
        {
            for(int j = 0; j < model.V; j++)
            {
                double diff = Math.abs(fc1.probs[t][j] - fc2.probs[t][j]);
                if(diff > max_diff_prior)
                    max_diff_prior = diff;
            }
        }

        double max_diff_subsequent = 0.0;
        for(int t = pos_modified; t < T; t++)
        {
            for(int j = 0; j < model.V; j++)
            {
                double diff = Math.abs(fc1.probs[t][j] - fc2.probs[t][j]);
                if(diff > max_diff_subsequent)
                    max_diff_subsequent = diff;
            }
        }

        System.out.printf("Causality Test: Max prob difference for pos 0..%d = %.2e\n", (pos_modified - 1), max_diff_prior);
        System.out.printf("Causality Test: Max prob difference for pos %d..%d = %.4f\n", pos_modified, (T - 1), max_diff_subsequent);

        boolean passed = (max_diff_prior < 1e-12) && (max_diff_subsequent > 1e-4);
        return passed;
    }

    // =========================================================================
    // VALIDATION EVALUATION
    // =========================================================================
    public static double[] evaluate_val_loss(String val_text, char[] vocab, TransformerModel model, int stride)
    {
        int T = model.T;
        int limit = val_text.length() - T - 1;
        double sum_all = 0.0;
        long count_all = 0;
        double sum_half = 0.0;
        long count_half = 0;

        int[] window = new int[T];
        int[] target = new int[T];

        for(int t = 0; t <= limit; t += stride)
        {
            for(int i = 0; i < T; i++)
            {
                window[i] = find_ch_number(val_text.charAt(t + i), vocab);
                target[i] = find_ch_number(val_text.charAt(t + i + 1), vocab);
            }
            ForwardCache fc = forward(window, target, model);
            for(int i = 0; i < T; i++)
            {
                sum_all += fc.pos_losses[i];
                count_all++;
                if(i >= 8)
                {
                    sum_half += fc.pos_losses[i];
                    count_half++;
                }
            }
        }
        return new double[]{ sum_all / count_all, sum_half / count_half };
    }

    // =========================================================================
    // TEXT GENERATION
    // =========================================================================
    public static String generate_text(TransformerModel model, String seed_prompt, int length_to_gen, char[] vocab)
    {
        int T = model.T;
        StringBuilder sb = new StringBuilder(seed_prompt);

        int[] window = new int[T];

        for(int step = 0; step < length_to_gen; step++)
        {
            int cur_len = sb.length();
            int start = Math.max(0, cur_len - T);
            String sub = sb.substring(start);

            // Left-pad if less than T characters
            int pad = T - sub.length();
            for(int i = 0; i < pad; i++)
                window[i] = 0; // space or 0
            for(int i = 0; i < sub.length(); i++)
                window[pad + i] = find_ch_number(sub.charAt(i), vocab);

            ForwardCache fc = forward(window, null, model);
            double[] last_probs = fc.probs[T - 1];

            // Sample from distribution
            double r = ThreadLocalRandom.current().nextDouble();
            double cum = 0.0;
            int picked = vocab.length - 1;
            for(int j = 0; j < vocab.length; j++)
            {
                cum += last_probs[j];
                if(r <= cum)
                {
                    picked = j;
                    break;
                }
            }

            sb.append(vocab[picked]);
        }

        return sb.substring(seed_prompt.length());
    }
}
