import java.util.Locale;

public class Driver 
{
    public static void main(String[] args) 
    {
        Locale.setDefault(Locale.US);

        String text = Data.chunk1;
        if(text == null || text.length() == 0)
        {
            System.out.println("Error: Text could not be loaded.");
            return;
        }

        Library.fill_the_characters_present_in_text(text);
        char[] vocab = Library.ch_present;
        int V = vocab.length;

        int split_idx = (int)(text.length() * 0.90);
        String train_text = text.substring(0, split_idx);
        String val_text = text.substring(split_idx);

        System.out.println("=================================================");
        System.out.println("STAGE 8: BOTTLENECK REMOVAL & MULTI-HEAD ATTENTION");
        System.out.println("Vocabulary size: " + V);
        System.out.println("Train characters: " + train_text.length() + ", Val characters: " + val_text.length());
        System.out.println("=================================================\n");

        // -------------------------------------------------------------
        // GRADIENT CHECKS (Wv, W1, and Wk after training steps)
        // -------------------------------------------------------------
        System.out.println("--- Corrected Gradient Checks (relative error < 0.001) ---");
        run_gradient_checks(train_text, vocab, V);
        System.out.println();

        int epochs = 5;

        // -------------------------------------------------------------
        // CHANGE 1: Bigger D (D = 32, N = 16, Joined residual)
        // -------------------------------------------------------------
        System.out.println("--- Change 1: Bigger D (D=32, N=16, Joined Input [att_out, x_N]) ---");
        run_change1(train_text, val_text, vocab, V, epochs);
        System.out.println();

        // -------------------------------------------------------------
        // CHANGE 2: Residual Connection (x_N + att_out, additive)
        // -------------------------------------------------------------
        System.out.println("--- Change 2: Additive Residual Connection (x_N + att_out) ---");
        run_change2(train_text, val_text, vocab, V, epochs);
        System.out.println();

        // -------------------------------------------------------------
        // CHANGE 3: Multi-Head Attention (4 heads x dk=8, Residual)
        // -------------------------------------------------------------
        System.out.println("--- Change 3: Multi-Head Attention (H=4 heads, dk=8, D=32, Additive Residual) ---");
        run_change3(train_text, val_text, vocab, V, epochs);
        System.out.println();
    }

    public static void run_gradient_checks(String train_text, char[] vocab, int V)
    {
        int N = 16;
        int D = 32;
        int H = 100;
        double lr = 0.01;
        double eps = 1e-5;

        double[][] C = Library.make_random_matrix(V, D, 0.05);
        double[][] P = Library.make_random_matrix(N, D, 0.05);
        double[][] Wq = Library.make_random_matrix(D, D, 0.05);
        double[][] Wk = Library.make_random_matrix(D, D, 0.05);
        double[][] Wv = Library.make_random_matrix(D, D, 0.05);
        double[][] W1 = Library.make_random_matrix(D, H, 0.05);
        double[] b1 = Library.make_zeros_vector(H);
        double[][] W2 = Library.make_random_matrix(H, V, 0.05);
        double[] b2 = Library.make_zeros_vector(V);

        // Train for 500 steps to make attention non-uniform
        for(int t = 0; t < 500; t++)
        {
            int[] ctx = new int[N];
            for(int i = 0; i < N; i++)
                ctx[i] = Library.find_ch_number(train_text.charAt(t + i), vocab);
            int target = Library.find_ch_number(train_text.charAt(t + N), vocab);

            Library.SingleHeadCache fc = Library.forward_single_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, H, V, true);
            Library.backward_single_head(fc, ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, lr, N, D, H, V, true);
        }

        // Test position
        int test_pos = 600;
        int[] ctx = new int[N];
        for(int i = 0; i < N; i++)
            ctx[i] = Library.find_ch_number(train_text.charAt(test_pos + i), vocab);
        int target = Library.find_ch_number(train_text.charAt(test_pos + N), vocab);

        // 1. Check Wv[2][3]
        int check_r = 2, check_c = 3;
        double orig_val = Wv[check_r][check_c];
        Wv[check_r][check_c] = orig_val + eps;
        double loss_plus = Library.forward_single_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, H, V, true).loss;
        Wv[check_r][check_c] = orig_val - eps;
        double loss_minus = Library.forward_single_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, H, V, true).loss;
        Wv[check_r][check_c] = orig_val;
        double num_grad_wv = (loss_plus - loss_minus) / (2.0 * eps);

        // Analytical gradient on Wv
        Library.SingleHeadCache fc = Library.forward_single_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, H, V, true);
        double[] g = new double[V];
        for(int j = 0; j < V; j++) g[j] = fc.probs[j] - ((j == target) ? 1.0 : 0.0);
        double[] dh = new double[H];
        for(int col = 0; col < H; col++) {
            double sum = 0.0;
            for(int j = 0; j < V; j++) sum += W2[col][j] * g[j];
            dh[col] = sum;
        }
        double[] dz = new double[H];
        for(int col = 0; col < H; col++) dz[col] = dh[col] * (1.0 - fc.h[col] * fc.h[col]);
        double[] d_net_in = new double[D];
        for(int row = 0; row < D; row++) {
            double sum = 0.0;
            for(int col = 0; col < H; col++) sum += W1[row][col] * dz[col];
            d_net_in[row] = sum;
        }
        double[] d_att_out = d_net_in; // residual additive
        double ana_grad_wv = 0.0;
        for(int i = 0; i < N; i++) {
            ana_grad_wv += fc.x[i][check_r] * fc.a[i] * d_att_out[check_c];
        }
        double rel_wv = Library.relative_error(ana_grad_wv, num_grad_wv);
        System.out.printf("Check Wv[2][3] : Analytic = %12.8f, Numerical = %12.8f, Rel Error = %.2e %s\n",
                ana_grad_wv, num_grad_wv, rel_wv, (rel_wv < 0.001 ? "[PASS]" : "[FAIL]"));

        // 2. Check W1[4][5]
        int w1_r = 4, w1_c = 5;
        double orig_w1 = W1[w1_r][w1_c];
        W1[w1_r][w1_c] = orig_w1 + eps;
        loss_plus = Library.forward_single_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, H, V, true).loss;
        W1[w1_r][w1_c] = orig_w1 - eps;
        loss_minus = Library.forward_single_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, H, V, true).loss;
        W1[w1_r][w1_c] = orig_w1;
        double num_grad_w1 = (loss_plus - loss_minus) / (2.0 * eps);
        double ana_grad_w1 = fc.net_in[w1_r] * dz[w1_c];
        double rel_w1 = Library.relative_error(ana_grad_w1, num_grad_w1);
        System.out.printf("Check W1[4][5] : Analytic = %12.8f, Numerical = %12.8f, Rel Error = %.2e %s\n",
                ana_grad_w1, num_grad_w1, rel_w1, (rel_w1 < 0.001 ? "[PASS]" : "[FAIL]"));

        // 3. Check Wk[1][2] after training steps
        int wk_r = 1, wk_c = 2;
        double orig_wk = Wk[wk_r][wk_c];
        Wk[wk_r][wk_c] = orig_wk + eps;
        loss_plus = Library.forward_single_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, H, V, true).loss;
        Wk[wk_r][wk_c] = orig_wk - eps;
        loss_minus = Library.forward_single_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, H, V, true).loss;
        Wk[wk_r][wk_c] = orig_wk;
        double num_grad_wk = (loss_plus - loss_minus) / (2.0 * eps);

        // Analytical gradient on Wk
        double[] da = new double[N];
        double sum_a_da = 0.0;
        for(int i = 0; i < N; i++) {
            double dot = 0.0;
            for(int m = 0; m < D; m++) dot += d_att_out[m] * fc.v[i][m];
            da[i] = dot;
            sum_a_da += fc.a[i] * da[i];
        }
        double[] ds = new double[N];
        for(int i = 0; i < N; i++) ds[i] = fc.a[i] * (da[i] - sum_a_da);
        double scale = Math.sqrt(D);
        double ana_grad_wk = 0.0;
        for(int i = 0; i < N; i++) {
            ana_grad_wk += fc.x[i][wk_r] * (ds[i] / scale) * fc.q[wk_c];
        }
        double rel_wk = Library.relative_error(ana_grad_wk, num_grad_wk);
        System.out.printf("Check Wk[1][2] : Analytic = %12.8f, Numerical = %12.8f, Rel Error = %.2e %s\n",
                ana_grad_wk, num_grad_wk, rel_wk, (rel_wk < 0.001 ? "[PASS]" : "[FAIL]"));
    }

    public static void run_change1(String train_text, String val_text, char[] vocab, int V, int epochs)
    {
        int N = 16;
        int D = 32;
        int H = 100;
        double lr = 0.01;

        double[][] C = Library.make_random_matrix(V, D, 0.05);
        double[][] P = Library.make_random_matrix(N, D, 0.05);
        double[][] Wq = Library.make_random_matrix(D, D, 0.05);
        double[][] Wk = Library.make_random_matrix(D, D, 0.05);
        double[][] Wv = Library.make_random_matrix(D, D, 0.05);
        double[][] W1 = Library.make_random_matrix(2 * D, H, 0.05);
        double[] b1 = Library.make_zeros_vector(H);
        double[][] W2 = Library.make_random_matrix(H, V, 0.05);
        double[] b2 = Library.make_zeros_vector(V);

        int train_len = train_text.length() - N;
        int val_len = val_text.length() - N;

        for(int epoch = 1; epoch <= epochs; epoch++)
        {
            double train_loss_sum = 0.0;
            for(int t = 0; t < train_len; t++)
            {
                int[] ctx = new int[N];
                for(int i = 0; i < N; i++)
                    ctx[i] = Library.find_ch_number(train_text.charAt(t + i), vocab);
                int target = Library.find_ch_number(train_text.charAt(t + N), vocab);

                Library.SingleHeadCache fc = Library.forward_single_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, H, V, false);
                Library.backward_single_head(fc, ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, lr, N, D, H, V, false);
                train_loss_sum += fc.loss;
            }

            double val_loss_sum = 0.0;
            for(int t = 0; t < val_len; t++)
            {
                int[] ctx = new int[N];
                for(int i = 0; i < N; i++)
                    ctx[i] = Library.find_ch_number(val_text.charAt(t + i), vocab);
                int target = Library.find_ch_number(val_text.charAt(t + N), vocab);

                Library.SingleHeadCache fc = Library.forward_single_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, H, V, false);
                val_loss_sum += fc.loss;
            }

            double avg_train = train_loss_sum / train_len;
            double avg_val = val_loss_sum / val_len;
            System.out.printf("Epoch %2d | Train Loss: %.4f | Val Loss: %.4f\n", epoch, avg_train, avg_val);
            lr *= 0.95;
        }
    }

    public static void run_change2(String train_text, String val_text, char[] vocab, int V, int epochs)
    {
        int N = 16;
        int D = 32;
        int H = 100;
        double lr = 0.01;

        double[][] C = Library.make_random_matrix(V, D, 0.05);
        double[][] P = Library.make_random_matrix(N, D, 0.05);
        double[][] Wq = Library.make_random_matrix(D, D, 0.05);
        double[][] Wk = Library.make_random_matrix(D, D, 0.05);
        double[][] Wv = Library.make_random_matrix(D, D, 0.05);
        double[][] W1 = Library.make_random_matrix(D, H, 0.05);
        double[] b1 = Library.make_zeros_vector(H);
        double[][] W2 = Library.make_random_matrix(H, V, 0.05);
        double[] b2 = Library.make_zeros_vector(V);

        int train_len = train_text.length() - N;
        int val_len = val_text.length() - N;

        for(int epoch = 1; epoch <= epochs; epoch++)
        {
            double train_loss_sum = 0.0;
            for(int t = 0; t < train_len; t++)
            {
                int[] ctx = new int[N];
                for(int i = 0; i < N; i++)
                    ctx[i] = Library.find_ch_number(train_text.charAt(t + i), vocab);
                int target = Library.find_ch_number(train_text.charAt(t + N), vocab);

                Library.SingleHeadCache fc = Library.forward_single_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, H, V, true);
                Library.backward_single_head(fc, ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, lr, N, D, H, V, true);
                train_loss_sum += fc.loss;
            }

            double val_loss_sum = 0.0;
            for(int t = 0; t < val_len; t++)
            {
                int[] ctx = new int[N];
                for(int i = 0; i < N; i++)
                    ctx[i] = Library.find_ch_number(val_text.charAt(t + i), vocab);
                int target = Library.find_ch_number(val_text.charAt(t + N), vocab);

                Library.SingleHeadCache fc = Library.forward_single_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, H, V, true);
                val_loss_sum += fc.loss;
            }

            double avg_train = train_loss_sum / train_len;
            double avg_val = val_loss_sum / val_len;
            System.out.printf("Epoch %2d | Train Loss: %.4f | Val Loss: %.4f\n", epoch, avg_train, avg_val);
            lr *= 0.95;
        }
    }

    public static void run_change3(String train_text, String val_text, char[] vocab, int V, int epochs)
    {
        int N = 16;
        int D = 32;
        int n_heads = 4;
        int head_dim = D / n_heads; // 8
        int H = 100;
        double lr = 0.01;

        double[][] C = Library.make_random_matrix(V, D, 0.05);
        double[][] P = Library.make_random_matrix(N, D, 0.05);
        double[][][] Wq = new double[n_heads][D][head_dim];
        double[][][] Wk = new double[n_heads][D][head_dim];
        double[][][] Wv = new double[n_heads][D][head_dim];
        for(int h = 0; h < n_heads; h++)
        {
            Wq[h] = Library.make_random_matrix(D, head_dim, 0.05);
            Wk[h] = Library.make_random_matrix(D, head_dim, 0.05);
            Wv[h] = Library.make_random_matrix(D, head_dim, 0.05);
        }

        double[][] W1 = Library.make_random_matrix(D, H, 0.05);
        double[] b1 = Library.make_zeros_vector(H);
        double[][] W2 = Library.make_random_matrix(H, V, 0.05);
        double[] b2 = Library.make_zeros_vector(V);

        int train_len = train_text.length() - N;
        int val_len = val_text.length() - N;

        for(int epoch = 1; epoch <= epochs; epoch++)
        {
            double train_loss_sum = 0.0;
            for(int t = 0; t < train_len; t++)
            {
                int[] ctx = new int[N];
                for(int i = 0; i < N; i++)
                    ctx[i] = Library.find_ch_number(train_text.charAt(t + i), vocab);
                int target = Library.find_ch_number(train_text.charAt(t + N), vocab);

                Library.MultiHeadCache mc = Library.forward_multi_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, n_heads, H, V);
                Library.backward_multi_head(mc, ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, lr, N, D, n_heads, H, V);
                train_loss_sum += mc.loss;
            }

            double val_loss_sum = 0.0;
            for(int t = 0; t < val_len; t++)
            {
                int[] ctx = new int[N];
                for(int i = 0; i < N; i++)
                    ctx[i] = Library.find_ch_number(val_text.charAt(t + i), vocab);
                int target = Library.find_ch_number(val_text.charAt(t + N), vocab);

                Library.MultiHeadCache mc = Library.forward_multi_head(ctx, target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, N, D, n_heads, H, V);
                val_loss_sum += mc.loss;
            }

            double avg_train = train_loss_sum / train_len;
            double avg_val = val_loss_sum / val_len;
            System.out.printf("Epoch %2d | Train Loss: %.4f | Val Loss: %.4f\n", epoch, avg_train, avg_val);
            lr *= 0.95;
        }
    }
}
