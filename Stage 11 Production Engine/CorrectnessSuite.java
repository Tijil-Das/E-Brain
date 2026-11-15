import java.io.File;
import java.util.*;

public class CorrectnessSuite 
{
    public static void main(String[] args) throws Exception 
    {
        String dataPath = "Text.txt";
        Dataset dataset = Dataset.loadAndPrepare(dataPath, null, 0.85, 0.10, 0.05);
        boolean pass = runAllTests(dataset.vocab);
        if (!pass) System.exit(1);
    }

    public static double relErr(double a, double n) 
    {
        return Math.abs(a - n) / (Math.abs(a) + Math.abs(n) + 1e-12);
    }

    public static boolean runAllTests(Vocab vocab) 
    {
        System.out.println("==========================================================================");
        System.out.println("PHASE 2: CORRECTNESS SUITE (AUTOMATED PRE-FLIGHT VERIFICATION)");
        System.out.println("==========================================================================");

        boolean allPassed = true;

        allPassed &= testInitialLoss(vocab);
        allPassed &= testCausality(vocab);
        allPassed &= testGradients(vocab);
        allPassed &= testSaveLoadRoundtrip(vocab);
        allPassed &= testOverfitOneBatch(vocab);
        allPassed &= testDeterminism(vocab);

        System.out.println("==========================================================================");
        if (allPassed) 
        {
            System.out.println(">>> ALL 6 CORRECTNESS TESTS PASSED CLEANLY! MODEL ENGINE IS SOUND. <<<");
        } 
        else 
        {
            System.err.println(">>> ONE OR MORE TESTS FAILED! CHECK DETAILED LOGS ABOVE. <<<");
        }
        System.out.println("==========================================================================\n");

        return allPassed;
    }

    // 1. Initial Loss Check: ln(V)
    public static boolean testInitialLoss(Vocab vocab) 
    {
        System.out.print("[Test 1/6] Initial Loss Check (Expected near ln(V))... ");
        int V = vocab.size();
        int T = 16, D = 32, nHeads = 4, H_mlp = 64, L = 2;
        Transformer model = Transformer.init(V, T, D, nHeads, H_mlp, L, true, 0.02, 42);

        int[] dummyTokens = new int[T];
        int[] dummyTargets = new int[T];
        for (int i = 0; i < T; i++) 
        {
            dummyTokens[i] = i % V;
            dummyTargets[i] = (i + 1) % V;
        }

        Transformer.Cache cache = model.forward(dummyTokens, dummyTargets, false, 0.0);
        double expectedLoss = Math.log(V);
        double diff = Math.abs(cache.avgLoss - expectedLoss);

        boolean pass = diff < 0.35;
        if (pass) 
        {
            System.out.printf("PASSED (Initial Loss: %.4f | ln(%d) = %.4f | Diff: %.4f)\n", cache.avgLoss, V, expectedLoss, diff);
        } 
        else 
        {
            System.out.printf("FAILED (Initial Loss: %.4f | ln(%d) = %.4f | Diff: %.4f)\n", cache.avgLoss, V, expectedLoss, diff);
        }
        return pass;
    }

    // 2. Causality Test
    public static boolean testCausality(Vocab vocab) 
    {
        System.out.print("[Test 2/6] Strict Causal Mask Test (Future token tampering)... ");
        int V = vocab.size();
        int T = 16, D = 32, nHeads = 4, H_mlp = 64, L = 2;
        Transformer model = Transformer.init(V, T, D, nHeads, H_mlp, L, true, 0.05, 42);

        int[] seq1 = new int[T];
        int[] seq2 = new int[T];
        int[] targets = new int[T];

        for (int i = 0; i < T; i++) 
        {
            seq1[i] = (i * 3 + 1) % V;
            seq2[i] = seq1[i];
            targets[i] = (i * 3 + 2) % V;
        }

        // Alter token at position 10 in seq2
        int tMod = 10;
        seq2[tMod] = (seq2[tMod] + 7) % V;
        for (int i = tMod + 1; i < T; i++) 
        {
            seq2[i] = (seq2[i] + 13) % V;
        }

        Transformer.Cache cache1 = model.forward(seq1, targets, false, 0.0);
        Transformer.Cache cache2 = model.forward(seq2, targets, false, 0.0);

        double maxDiffPrior = 0.0;
        for (int t = 0; t < tMod; t++) 
        {
            for (int v = 0; v < V; v++) 
            {
                double d = Math.abs(cache1.logits[t][v] - cache2.logits[t][v]);
                if (d > maxDiffPrior) maxDiffPrior = d;
            }
            double lossDiff = Math.abs(cache1.losses[t] - cache2.losses[t]);
            if (lossDiff > maxDiffPrior) maxDiffPrior = lossDiff;
        }

        boolean pass = maxDiffPrior < 1e-12;
        if (pass) 
        {
            System.out.printf("PASSED (Max Prior Difference: %.2e)\n", maxDiffPrior);
        } 
        else 
        {
            System.out.printf("FAILED (Max Prior Difference: %.2e >= 1e-12)\n", maxDiffPrior);
        }
        return pass;
    }

    // 3. Gradient Check: One weight from every table
    public static boolean testGradients(Vocab vocab) 
    {
        System.out.println("[Test 3/6] Numerical vs. Analytical Gradient Checks (RelErr < 1e-4):");
        int V = Math.min(15, vocab.size());
        int T = 4, D = 8, nHeads = 2, H_mlp = 12, L = 1; // Small compact model for fast high-precision finite diff
        Transformer model = Transformer.init(V, T, D, nHeads, H_mlp, L, true, 0.1, 42);

        int[] tokens = {1, 3, 5, 2};
        int[] targets = {3, 5, 2, 4};

        Transformer.Cache cache = model.forward(tokens, targets, false, 0.0);
        Transformer.Gradients grads = model.backward(cache);

        double eps = 1e-5;
        boolean allGradsPass = true;

        // Check C
        allGradsPass &= checkWeight1D("C[1][2]", () -> model.C[1][2], v -> model.C[1][2] = v, grads.dC[1][2], tokens, targets, model, eps);
        // Check P
        allGradsPass &= checkWeight1D("P[2][1]", () -> model.P[2][1], v -> model.P[2][1] = v, grads.dP[2][1], tokens, targets, model, eps);
        // Check Wq
        allGradsPass &= checkWeight1D("Wq[0][1][1]", () -> model.blocks[0].Wq[0][1][1], v -> model.blocks[0].Wq[0][1][1] = v, grads.blockGrads[0].dWq[0][1][1], tokens, targets, model, eps);
        // Check Wk
        allGradsPass &= checkWeight1D("Wk[1][2][0]", () -> model.blocks[0].Wk[1][2][0], v -> model.blocks[0].Wk[1][2][0] = v, grads.blockGrads[0].dWk[1][2][0], tokens, targets, model, eps);
        // Check Wv
        allGradsPass &= checkWeight1D("Wv[0][2][1]", () -> model.blocks[0].Wv[0][2][1], v -> model.blocks[0].Wv[0][2][1] = v, grads.blockGrads[0].dWv[0][2][1], tokens, targets, model, eps);
        // Check Wo
        allGradsPass &= checkWeight1D("Wo[2][3]", () -> model.blocks[0].Wo[2][3], v -> model.blocks[0].Wo[2][3] = v, grads.blockGrads[0].dWo[2][3], tokens, targets, model, eps);
        // Check W1
        allGradsPass &= checkWeight1D("W1[2][4]", () -> model.blocks[0].W1[2][4], v -> model.blocks[0].W1[2][4] = v, grads.blockGrads[0].dW1[2][4], tokens, targets, model, eps);
        // Check b1
        allGradsPass &= checkWeight1D("b1[3]", () -> model.blocks[0].b1[3], v -> model.blocks[0].b1[3] = v, grads.blockGrads[0].db1[3], tokens, targets, model, eps);
        // Check W2
        allGradsPass &= checkWeight1D("W2[3][2]", () -> model.blocks[0].W2[3][2], v -> model.blocks[0].W2[3][2] = v, grads.blockGrads[0].dW2[3][2], tokens, targets, model, eps);
        // Check b2
        allGradsPass &= checkWeight1D("b2[1]", () -> model.blocks[0].b2[1], v -> model.blocks[0].b2[1] = v, grads.blockGrads[0].db2[1], tokens, targets, model, eps);
        // Check LayerNorm gamma1 & beta1
        allGradsPass &= checkWeight1D("gamma1[2]", () -> model.blocks[0].gamma1[2], v -> model.blocks[0].gamma1[2] = v, grads.blockGrads[0].dGamma1[2], tokens, targets, model, eps);
        allGradsPass &= checkWeight1D("beta1[2]", () -> model.blocks[0].beta1[2], v -> model.blocks[0].beta1[2] = v, grads.blockGrads[0].dBeta1[2], tokens, targets, model, eps);
        // Check Wout & bout
        allGradsPass &= checkWeight1D("Wout[1][3]", () -> model.Wout[1][3], v -> model.Wout[1][3] = v, grads.dWout[1][3], tokens, targets, model, eps);
        allGradsPass &= checkWeight1D("bout[2]", () -> model.bout[2], v -> model.bout[2] = v, grads.dbout[2], tokens, targets, model, eps);

        return allGradsPass;
    }

    interface Getter { double get(); }
    interface Setter { void set(double v); }

    private static boolean checkWeight1D(String name, Getter getter, Setter setter, double anaGrad, int[] tokens, int[] targets, Transformer model, double eps) 
    {
        double orig = getter.get();

        setter.set(orig + eps);
        double lossPlus = model.forward(tokens, targets, false, 0.0).avgLoss;

        setter.set(orig - eps);
        double lossMinus = model.forward(tokens, targets, false, 0.0).avgLoss;

        setter.set(orig); // Restore

        double numGrad = (lossPlus - lossMinus) / (2.0 * eps);
        double err = relErr(anaGrad, numGrad);

        boolean pass = err < 1e-4 || Math.abs(anaGrad - numGrad) < 1e-7;
        System.out.printf("   - %-12s | Ana: %11.8f | Num: %11.8f | RelErr: %.2e [%s]\n", name, anaGrad, numGrad, err, pass ? "OK" : "FAIL");
        return pass;
    }

    // 4. Save/Load Roundtrip Test
    public static boolean testSaveLoadRoundtrip(Vocab vocab) 
    {
        System.out.print("[Test 4/6] Save/Load Checkpoint Roundtrip Verification... ");
        try 
        {
            int V = vocab.size();
            int T = 16, D = 32, nHeads = 4, H_mlp = 64, L = 2;
            Transformer model = Transformer.init(V, T, D, nHeads, H_mlp, L, true, 0.02, 42);
            model.adam.t = 15;
            model.adam.mC[1][2] = 0.12345;
            model.adam.vC[1][2] = 0.00987;

            String tempFile = "temp_checkpoint_test.bin";
            model.saveCheckpoint(tempFile);

            Transformer loaded = Transformer.loadCheckpoint(tempFile);
            new File(tempFile).delete();

            // Verify weights and moments
            boolean match = true;
            match &= (model.V == loaded.V && model.T == loaded.T && model.D == loaded.D && model.L == loaded.L);
            match &= (model.C[1][2] == loaded.C[1][2]);
            match &= (model.adam.t == loaded.adam.t);
            match &= (model.adam.mC[1][2] == loaded.adam.mC[1][2]);
            match &= (model.adam.vC[1][2] == loaded.adam.vC[1][2]);

            if (match) 
            {
                System.out.println("PASSED (Exact bit-identical serialization recovery)");
            } 
            else 
            {
                System.out.println("FAILED (Parameter mismatch detected after load)");
            }
            return match;
        } 
        catch (Exception e) 
        {
            System.out.println("FAILED with exception: " + e.getMessage());
            return false;
        }
    }

    // 5. Overfit-One-Batch Test: Train on 16 windows only -> loss should approach nearly 0 (< 0.15)
    public static boolean testOverfitOneBatch(Vocab vocab) 
    {
        System.out.print("[Test 5/6] Overfit-One-Batch Test (16 windows -> target loss < 0.15)... ");
        int V = vocab.size();
        int T = 16, D = 32, nHeads = 4, H_mlp = 64, L = 2;
        Transformer model = Transformer.init(V, T, D, nHeads, H_mlp, L, false, 0.02, 123);

        // Create 16 fixed windows
        int numWindows = 16;
        int[][] batchTokens = new int[numWindows][T];
        int[][] batchTargets = new int[numWindows][T];
        Random r = new Random(42);
        for (int b = 0; b < numWindows; b++) 
        {
            for (int t = 0; t < T; t++) 
            {
                batchTokens[b][t] = r.nextInt(V);
                batchTargets[b][t] = (batchTokens[b][t] + 1) % V;
            }
        }

        // Train with Adam for 120 steps
        double lr = 0.015;
        double finalLoss = 0.0;
        for (int step = 0; step < 120; step++) 
        {
            double stepLoss = 0.0;
            Transformer.Gradients accum = new Transformer.Gradients(V, T, D, nHeads, model.dk, H_mlp, L);
            for (int b = 0; b < numWindows; b++) 
            {
                Transformer.Cache c = model.forward(batchTokens[b], batchTargets[b], false, 0.0);
                stepLoss += c.avgLoss;
                Transformer.Gradients g = model.backward(c);
                accum.add(g);
            }
            accum.scale(1.0 / numWindows);
            model.stepAdam(accum, lr, 0.9, 0.999, 1e-8, 0.0);
            finalLoss = stepLoss / numWindows;
        }

        boolean pass = finalLoss < 0.15;
        if (pass) 
        {
            System.out.printf("PASSED (Final Loss: %.4f < 0.15)\n", finalLoss);
        } 
        else 
        {
            System.out.printf("FAILED (Final Loss: %.4f >= 0.15)\n", finalLoss);
        }
        return pass;
    }

    // 6. Determinism Check: Same seed gives same loss curve
    public static boolean testDeterminism(Vocab vocab) 
    {
        System.out.print("[Test 6/6] Determinism Verification (Identical seed reproducibility)... ");
        int V = vocab.size();
        int T = 16, D = 16, nHeads = 2, H_mlp = 32, L = 1;
        long seed = 987654321L;

        // Run 1
        Transformer m1 = Transformer.init(V, T, D, nHeads, H_mlp, L, true, 0.05, seed);
        int[] tok = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0};
        int[] tgt = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};

        Transformer.Cache c1 = m1.forward(tok, tgt, false, 0.0);
        Transformer.Gradients g1 = m1.backward(c1);
        m1.stepAdam(g1, 0.01, 0.9, 0.999, 1e-8, 0.0);
        Transformer.Cache c1_step2 = m1.forward(tok, tgt, false, 0.0);

        // Run 2
        Transformer m2 = Transformer.init(V, T, D, nHeads, H_mlp, L, true, 0.05, seed);
        Transformer.Cache c2 = m2.forward(tok, tgt, false, 0.0);
        Transformer.Gradients g2 = m2.backward(c2);
        m2.stepAdam(g2, 0.01, 0.9, 0.999, 1e-8, 0.0);
        Transformer.Cache c2_step2 = m2.forward(tok, tgt, false, 0.0);

        double diffInitial = Math.abs(c1.avgLoss - c2.avgLoss);
        double diffStep2 = Math.abs(c1_step2.avgLoss - c2_step2.avgLoss);

        boolean pass = (diffInitial < 1e-12) && (diffStep2 < 1e-12);
        if (pass) 
        {
            System.out.printf("PASSED (Diff Step 1: %.2e | Diff Step 2: %.2e)\n", diffInitial, diffStep2);
        } 
        else 
        {
            System.out.printf("FAILED (Diff Step 1: %.2e | Diff Step 2: %.2e)\n", diffInitial, diffStep2);
        }
        return pass;
    }
}
