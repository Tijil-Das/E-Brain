public class TestGrad 
{
    public static void main(String[] args) 
    {
        String text = Data.chunk1;
        TransformerLibrary.fill_the_characters_present_in_text(text);
        char[] vocab = TransformerLibrary.ch_present;
        int V = vocab.length;
        int T = 16, D = 32, n_heads = 4, H_mlp = 100, L = 1;

        TransformerLibrary.TransformerModel model = TransformerLibrary.TransformerModel.init(V, T, D, n_heads, H_mlp, L, 0.05);

        // Train 600 steps to make attention non-uniform
        int[] window = new int[T];
        int[] target = new int[T];
        for(int step = 0; step < 600; step++)
        {
            for(int i = 0; i < T; i++)
            {
                window[i] = TransformerLibrary.find_ch_number(text.charAt(step + i), vocab);
                target[i] = TransformerLibrary.find_ch_number(text.charAt(step + i + 1), vocab);
            }
            TransformerLibrary.ForwardCache fc = TransformerLibrary.forward(window, target, model);
            TransformerLibrary.backward(fc, window, target, model, 0.015);
        }

        System.out.println("--- Causality Test ---");
        boolean passed_causality = TransformerLibrary.run_causality_test(model, vocab);
        System.out.println("Causality Test Passed: " + passed_causality + "\n");

        int pos = 800;
        for(int i = 0; i < T; i++)
        {
            window[i] = TransformerLibrary.find_ch_number(text.charAt(pos + i), vocab);
            target[i] = TransformerLibrary.find_ch_number(text.charAt(pos + i + 1), vocab);
        }

        double eps = 1e-5;

        // 1. Check Wv[0][2][3]
        TransformerLibrary.BlockWeights bw = model.blocks[0];
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
        System.out.printf("Check Wv[0][2][3] : Analytic = %12.8f, Numerical = %12.8f, Rel Error = %.2e %s\n",
                ana_grad_wv, num_grad_wv, rel_wv, (rel_wv < 0.001 ? "[PASS]" : "[FAIL]"));

        // 2. Check W1[4][5]
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
        System.out.printf("Check W1[4][5]    : Analytic = %12.8f, Numerical = %12.8f, Rel Error = %.2e %s\n",
                ana_grad_w1, num_grad_w1, rel_w1, (rel_w1 < 0.001 ? "[PASS]" : "[FAIL]"));

        // 3. Find and check Wk element with non-trivial gradient
        copy_m = model.copy();
        fc = TransformerLibrary.forward(window, target, copy_m);
        TransformerLibrary.backward(fc, window, target, copy_m, 1.0);

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

        System.out.printf("Check Wk[%d][%d][%d] : Analytic = %12.8f, Numerical = %12.8f, Rel Error = %.2e %s\n",
                best_h, best_r, best_c, ana_grad_wk, num_grad_wk, rel_wk, (rel_wk < 0.001 ? "[PASS]" : "[FAIL]"));
    }
}
