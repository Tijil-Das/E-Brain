import java.util.Locale;

public class Driver 
{
    public static void main(String[] args) 
    {
        Locale.setDefault(Locale.US);

        String raw_text = Data.chunk1;
        if(raw_text == null || raw_text.length() == 0)
        {
            System.out.println("Error: Text could not be loaded.");
            return;
        }

        TransformerLibrary.fill_the_characters_present_in_text(raw_text);
        char[] vocab = TransformerLibrary.ch_present;
        int V = vocab.length;

        // 2x text
        String full_text = raw_text + raw_text;
        int split_idx = (int)(full_text.length() * 0.90);
        String train_text = full_text.substring(0, split_idx);
        String val_text = full_text.substring(split_idx);

        System.out.println("==========================================================================");
        System.out.println("STAGE 10: CAUSAL TRANSFORMER — PREDICTING AT EVERY POSITION");
        System.out.println("Vocabulary size: " + V);
        System.out.println("Train characters: " + train_text.length() + ", Val characters: " + val_text.length());
        System.out.println("==========================================================================\n");

        int T = 16;
        int D = 32;
        int n_heads = 4;
        int H_mlp = 100;

        // -------------------------------------------------------------
        // 1. CAUSALITY TEST & GRADIENT CHECKS
        // -------------------------------------------------------------
        System.out.println("--- 1. Verification Tests (Before Training) ---");
        TransformerLibrary.TransformerModel check_m = TransformerLibrary.TransformerModel.init(V, T, D, n_heads, H_mlp, 1, 0.05);

        // Train 600 steps for gradient check
        int[] w_tmp = new int[T];
        int[] t_tmp = new int[T];
        for(int step = 0; step < 600; step++)
        {
            for(int i = 0; i < T; i++)
            {
                w_tmp[i] = TransformerLibrary.find_ch_number(train_text.charAt(step + i), vocab);
                t_tmp[i] = TransformerLibrary.find_ch_number(train_text.charAt(step + i + 1), vocab);
            }
            TransformerLibrary.ForwardCache fc = TransformerLibrary.forward(w_tmp, t_tmp, check_m);
            TransformerLibrary.backward(fc, w_tmp, t_tmp, check_m, 0.015);
        }

        boolean causality_passed = TransformerLibrary.run_causality_test(check_m, vocab);
        System.out.println("Causality Test: " + (causality_passed ? "[PASSED - Future positions completely masked]" : "[FAILED]") + "\n");

        System.out.println("Gradient Checks (Relative Error < 0.001):");
        run_grad_checks(check_m, train_text, vocab, T, D, n_heads);
        System.out.println();

        int epochs = 12;
        int train_stride = 4;
        int val_stride = 4;
        double start_lr = 0.015;
        double lr_decay = 0.90;

        // -------------------------------------------------------------
        // 2. STAGE 10A: SINGLE BLOCK TRANSFORMER (L = 1)
        // -------------------------------------------------------------
        System.out.println("--- 2. Stage 10A: Single Block Causal Transformer (L = 1) ---");
        TransformerLibrary.TransformerModel model_10A = TransformerLibrary.TransformerModel.init(V, T, D, n_heads, H_mlp, 1, 0.05);
        TrainResult res_10A = train_model("10A (1 Block)", model_10A, train_text, val_text, vocab, epochs, train_stride, val_stride, start_lr, lr_decay);
        System.out.println();

        // -------------------------------------------------------------
        // 3. STAGE 10B: TWO STACKED BLOCKS TRANSFORMER (L = 2)
        // -------------------------------------------------------------
        System.out.println("--- 3. Stage 10B: Two Stacked Blocks Causal Transformer (L = 2) ---");
        TransformerLibrary.TransformerModel model_10B = TransformerLibrary.TransformerModel.init(V, T, D, n_heads, H_mlp, 2, 0.05);
        TrainResult res_10B = train_model("10B (2 Blocks)", model_10B, train_text, val_text, vocab, epochs, train_stride, val_stride, start_lr, lr_decay);
        System.out.println();

        // -------------------------------------------------------------
        // 4. SUMMARY COMPARISON & 300 CHAR GENERATION
        // -------------------------------------------------------------
        System.out.println("==========================================================================");
        System.out.println("STAGE 10 FINAL COMPARISON SUMMARY");
        System.out.println("Non-Attention Stage 6A Baseline : ~1.8596");
        System.out.printf("Stage 10A (1 Block)  : Best Val (All Pos 0..15) = %.4f | Best Val (Pos 8..15) = %.4f\n",
                res_10A.best_val_all, res_10A.best_val_half);
        System.out.printf("Stage 10B (2 Blocks) : Best Val (All Pos 0..15) = %.4f | Best Val (Pos 8..15) = %.4f\n",
                res_10B.best_val_all, res_10B.best_val_half);
        System.out.println("==========================================================================\n");

        // Pick best model for generation
        TransformerLibrary.TransformerModel best_overall = (res_10B.best_val_half < res_10A.best_val_half) ? res_10B.best_weights : res_10A.best_weights;
        String best_name = (res_10B.best_val_half < res_10A.best_val_half) ? "Stage 10B (2 Blocks)" : "Stage 10A (1 Block)";

        System.out.println("--- 300 Generated Characters from Best Model (" + best_name + ") ---");
        String seed = "Arthur Pendelton";
        String sample = TransformerLibrary.generate_text(best_overall, seed, 300, vocab);
        System.out.println(seed + sample);
        System.out.println("--------------------------------------------------------------------------");
    }

    public static class TrainResult
    {
        public double best_val_all;
        public double best_val_half;
        public int best_epoch;
        public TransformerLibrary.TransformerModel best_weights;
    }

    public static TrainResult train_model(String name, TransformerLibrary.TransformerModel model,
                                         String train_text, String val_text, char[] vocab,
                                         int epochs, int train_stride, int val_stride,
                                         double start_lr, double lr_decay)
    {
        TrainResult tr = new TrainResult();
        tr.best_val_all = Double.MAX_VALUE;
        tr.best_val_half = Double.MAX_VALUE;
        tr.best_epoch = 1;
        tr.best_weights = model.copy();

        int T = model.T;
        int limit = train_text.length() - T - 1;
        int[] window = new int[T];
        int[] target = new int[T];
        double current_lr = start_lr;

        for(int epoch = 1; epoch <= epochs; epoch++)
        {
            long t0 = System.currentTimeMillis();
            double train_loss_sum = 0.0;
            int n_windows = 0;

            for(int t = 0; t <= limit; t += train_stride)
            {
                for(int i = 0; i < T; i++)
                {
                    window[i] = TransformerLibrary.find_ch_number(train_text.charAt(t + i), vocab);
                    target[i] = TransformerLibrary.find_ch_number(train_text.charAt(t + i + 1), vocab);
                }
                TransformerLibrary.ForwardCache fc = TransformerLibrary.forward(window, target, model);
                TransformerLibrary.backward(fc, window, target, model, current_lr);
                train_loss_sum += fc.total_loss;
                n_windows++;
            }

            double avg_train = train_loss_sum / n_windows;
            double[] val_scores = TransformerLibrary.evaluate_val_loss(val_text, vocab, model, val_stride);
            double val_all = val_scores[0];
            double val_half = val_scores[1];
            long elapsed = System.currentTimeMillis() - t0;

            boolean is_best = false;
            if(val_half < tr.best_val_half)
            {
                tr.best_val_half = val_half;
                tr.best_val_all = val_all;
                tr.best_epoch = epoch;
                tr.best_weights = model.copy();
                is_best = true;
            }

            System.out.printf("Epoch %2d/%2d | Train: %.4f | Val (All 0..15): %.4f | Val (Pos 8..15): %.4f | %4d ms %s\n",
                    epoch, epochs, avg_train, val_all, val_half, elapsed, (is_best ? "<- [BEST]" : ""));

            current_lr *= lr_decay;
        }

        System.out.printf("[%s Summary] Best Val (Pos 8..15): %.4f (All Pos: %.4f) at Epoch %d\n",
                name, tr.best_val_half, tr.best_val_all, tr.best_epoch);
        return tr;
    }

    public static void run_grad_checks(TransformerLibrary.TransformerModel model, String train_text, char[] vocab,
                                       int T, int D, int n_heads)
    {
        int pos = 800;
        int[] window = new int[T];
        int[] target = new int[T];
        for(int i = 0; i < T; i++)
        {
            window[i] = TransformerLibrary.find_ch_number(train_text.charAt(pos + i), vocab);
            target[i] = TransformerLibrary.find_ch_number(train_text.charAt(pos + i + 1), vocab);
        }

        double eps = 1e-5;
        TransformerLibrary.BlockWeights bw = model.blocks[0];

        // 1. Wv check
        double orig_wv = bw.Wv[0][2][3];
        bw.Wv[0][2][3] = orig_wv + eps;
        double l_plus = TransformerLibrary.forward(window, target, model).total_loss;
        bw.Wv[0][2][3] = orig_wv - eps;
        double l_minus = TransformerLibrary.forward(window, target, model).total_loss;
        bw.Wv[0][2][3] = orig_wv;
        double num_grad_wv = (l_plus - l_minus) / (2.0 * eps);

        TransformerLibrary.TransformerModel copy_m = model.copy();
        TransformerLibrary.ForwardCache fc = TransformerLibrary.forward(window, target, copy_m);
        TransformerLibrary.backward(fc, window, target, copy_m, 1.0);
        double ana_grad_wv = orig_wv - copy_m.blocks[0].Wv[0][2][3];
        double rel_wv = TransformerLibrary.relative_error(ana_grad_wv, num_grad_wv);
        System.out.printf("  Check Wv[0][2][3] : Analytic = %12.8f, Numerical = %12.8f, Rel Error = %.2e %s\n",
                ana_grad_wv, num_grad_wv, rel_wv, (rel_wv < 0.001 ? "[PASS]" : "[FAIL]"));

        // 2. W1 check
        double orig_w1 = bw.W1[4][5];
        bw.W1[4][5] = orig_w1 + eps;
        l_plus = TransformerLibrary.forward(window, target, model).total_loss;
        bw.W1[4][5] = orig_w1 - eps;
        l_minus = TransformerLibrary.forward(window, target, model).total_loss;
        bw.W1[4][5] = orig_w1;
        double num_grad_w1 = (l_plus - l_minus) / (2.0 * eps);

        copy_m = model.copy();
        fc = TransformerLibrary.forward(window, target, copy_m);
        TransformerLibrary.backward(fc, window, target, copy_m, 1.0);
        double ana_grad_w1 = orig_w1 - copy_m.blocks[0].W1[4][5];
        double rel_w1 = TransformerLibrary.relative_error(ana_grad_w1, num_grad_w1);
        System.out.printf("  Check W1[4][5]    : Analytic = %12.8f, Numerical = %12.8f, Rel Error = %.2e %s\n",
                ana_grad_w1, num_grad_w1, rel_w1, (rel_w1 < 0.001 ? "[PASS]" : "[FAIL]"));

        // 3. Wk check
        int best_h = 0, best_r = 0, best_c = 0;
        double max_abs_wk = 0.0;
        for(int h = 0; h < n_heads; h++) {
            for(int r = 0; r < D; r++) {
                for(int c = 0; c < (D/n_heads); c++) {
                    double g_val = Math.abs(bw.Wk[h][r][c] - copy_m.blocks[0].Wk[h][r][c]);
                    if(g_val > max_abs_wk) {
                        max_abs_wk = g_val;
                        best_h = h; best_r = r; best_c = c;
                    }
                }
            }
        }

        double orig_wk = bw.Wk[best_h][best_r][best_c];
        bw.Wk[best_h][best_r][best_c] = orig_wk + eps;
        l_plus = TransformerLibrary.forward(window, target, model).total_loss;
        bw.Wk[best_h][best_r][best_c] = orig_wk - eps;
        l_minus = TransformerLibrary.forward(window, target, model).total_loss;
        bw.Wk[best_h][best_r][best_c] = orig_wk;
        double num_grad_wk = (l_plus - l_minus) / (2.0 * eps);
        double ana_grad_wk = orig_wk - copy_m.blocks[0].Wk[best_h][best_r][best_c];
        double rel_wk = TransformerLibrary.relative_error(ana_grad_wk, num_grad_wk);

        System.out.printf("  Check Wk[%d][%d][%d] : Analytic = %12.8f, Numerical = %12.8f, Rel Error = %.2e %s\n",
                best_h, best_r, best_c, ana_grad_wk, num_grad_wk, rel_wk, (rel_wk < 0.001 ? "[PASS]" : "[FAIL]"));
    }
}
