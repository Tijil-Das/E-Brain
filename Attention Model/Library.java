import java.util.concurrent.ThreadLocalRandom;

public class Library 
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

    //Step 1 to 5: Hand check verification (D=2, N=3, skip sqrt(D))
    public static double[][] run_hand_check()
    {
        int d = 2;
        int n = 3;
        double[] q = {1.0, 0.0};
        double[][] k = {
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 1.0}
        };
        double[][] v = {
            {2.0, 0.0},
            {0.0, 4.0},
            {1.0, 1.0}
        };

        //Scores = q . k_i
        double[] scores = new double[n];
        for(int i = 0; i < n; i++)
        {
            double dot = 0.0;
            for(int m = 0; m < d; m++)
                dot = dot + (q[m] * k[i][m]);
            scores[i] = dot;
        }

        //Softmax
        double max_s = scores[0];
        for(int i = 1; i < n; i++)
            if(scores[i] > max_s)
                max_s = scores[i];

        double[] exp_s = new double[n];
        double sum_exp = 0.0;
        for(int i = 0; i < n; i++)
        {
            exp_s[i] = Math.exp(scores[i] - max_s);
            sum_exp = sum_exp + exp_s[i];
        }

        double[] a = new double[n];
        for(int i = 0; i < n; i++)
            a[i] = exp_s[i] / sum_exp;

        //Blend = sum(a_i * v_i)
        double[] output = new double[d];
        for(int m = 0; m < d; m++)
        {
            double sum = 0.0;
            for(int i = 0; i < n; i++)
                sum = sum + (a[i] * v[i][m]);
            output[m] = sum;
        }

        return new double[][]{scores, a, output};
    }

    //Parameter initializers
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

    //Forward pass data container
    public static class ForwardCache
    {
        public double[][] x;       // N x D
        public double[] q;         // D
        public double[][] k;       // N x D
        public double[][] v;       // N x D
        public double[] scores;    // N
        public double[] a;         // N
        public double[] att_out;   // D
        public double[] net_in;    // 2*D (att_out joined with x_N)
        public double[] z;         // H
        public double[] h;         // H
        public double[] out_scores;// V
        public double[] probs;     // V
        public double loss;
    }

    public static ForwardCache forward(int[] context, int target_idx,
                                       double[][] C, double[][] P,
                                       double[][] Wq, double[][] Wk, double[][] Wv,
                                       double[][] W1, double[] b1, double[][] W2, double[] b2,
                                       int n, int d, int h_size, int v_size)
    {
        ForwardCache fc = new ForwardCache();
        double scale = Math.sqrt(d);

        //Step 1: Position vectors added to character embeddings: x_i = C[c_i] + P[i]
        fc.x = new double[n][d];
        for(int i = 0; i < n; i++)
        {
            int char_row = context[i];
            for(int m = 0; m < d; m++)
                fc.x[i][m] = C[char_row][m] + P[i][m];
        }

        //Step 2: Projections
        //Query q = x[n-1] * Wq
        fc.q = new double[d];
        for(int col = 0; col < d; col++)
        {
            double sum = 0.0;
            for(int row = 0; row < d; row++)
                sum = sum + (fc.x[n - 1][row] * Wq[row][col]);
            fc.q[col] = sum;
        }

        //Key k_i = x_i * Wk, Value v_i = x_i * Wv
        fc.k = new double[n][d];
        fc.v = new double[n][d];
        for(int i = 0; i < n; i++)
        {
            for(int col = 0; col < d; col++)
            {
                double sum_k = 0.0;
                double sum_v = 0.0;
                for(int row = 0; row < d; row++)
                {
                    sum_k = sum_k + (fc.x[i][row] * Wk[row][col]);
                    sum_v = sum_v + (fc.x[i][row] * Wv[row][col]);
                }
                fc.k[i][col] = sum_k;
                fc.v[i][col] = sum_v;
            }
        }

        //Step 3: Scores = (q . k_i) / sqrt(D)
        fc.scores = new double[n];
        for(int i = 0; i < n; i++)
        {
            double dot = 0.0;
            for(int m = 0; m < d; m++)
                dot = dot + (fc.q[m] * fc.k[i][m]);
            fc.scores[i] = dot / scale;
        }

        //Step 4: Softmax the N scores into attention weights a_i
        double max_s = fc.scores[0];
        for(int i = 1; i < n; i++)
            if(fc.scores[i] > max_s)
                max_s = fc.scores[i];

        double[] exp_s = new double[n];
        double sum_exp = 0.0;
        for(int i = 0; i < n; i++)
        {
            exp_s[i] = Math.exp(fc.scores[i] - max_s);
            sum_exp = sum_exp + exp_s[i];
        }

        fc.a = new double[n];
        for(int i = 0; i < n; i++)
            fc.a[i] = exp_s[i] / sum_exp;

        //Step 5: Blend output = sum(a_i * v_i)
        fc.att_out = new double[d];
        for(int m = 0; m < d; m++)
        {
            double sum = 0.0;
            for(int i = 0; i < n; i++)
                sum = sum + (fc.a[i] * fc.v[i][m]);
            fc.att_out[m] = sum;
        }

        //Step 6: Join with x[n-1] for residual connection (size = 2*D)
        fc.net_in = new double[2 * d];
        for(int m = 0; m < d; m++)
        {
            fc.net_in[m] = fc.att_out[m];
            fc.net_in[d + m] = fc.x[n - 1][m];
        }

        //Hidden layer: h = tanh(net_in * W1 + b1)
        fc.z = new double[h_size];
        fc.h = new double[h_size];
        for(int col = 0; col < h_size; col++)
        {
            double sum = 0.0;
            for(int row = 0; row < 2 * d; row++)
                sum = sum + (fc.net_in[row] * W1[row][col]);
            fc.z[col] = sum + b1[col];
            fc.h[col] = Math.tanh(fc.z[col]);
        }

        //Output layer: scores = h * W2 + b2
        fc.out_scores = new double[v_size];
        for(int col = 0; col < v_size; col++)
        {
            double sum = 0.0;
            for(int row = 0; row < h_size; row++)
                sum = sum + (fc.h[row] * W2[row][col]);
            fc.out_scores[col] = sum + b2[col];
        }

        //Softmax on V output scores
        double max_out = fc.out_scores[0];
        for(int j = 1; j < v_size; j++)
            if(fc.out_scores[j] > max_out)
                max_out = fc.out_scores[j];

        double[] exp_out = new double[v_size];
        double sum_exp_out = 0.0;
        for(int j = 0; j < v_size; j++)
        {
            exp_out[j] = Math.exp(fc.out_scores[j] - max_out);
            sum_exp_out = sum_exp_out + exp_out[j];
        }

        fc.probs = new double[v_size];
        for(int j = 0; j < v_size; j++)
            fc.probs[j] = exp_out[j] / sum_exp_out;

        fc.loss = -Math.log(fc.probs[target_idx]);
        return fc;
    }

    //Backward pass through Attention + Hidden Layer + Embeddings
    public static void backward_and_nudge(ForwardCache fc, int[] context, int target_idx,
                                          double[][] C, double[][] P,
                                          double[][] Wq, double[][] Wk, double[][] Wv,
                                          double[][] W1, double[] b1, double[][] W2, double[] b2,
                                          double lr, int n, int d, int h_size, int v_size)
    {
        double scale = Math.sqrt(d);

        //1. Output layer gradients
        double[] g = new double[v_size];
        for(int j = 0; j < v_size; j++)
            g[j] = fc.probs[j] - ((j == target_idx) ? 1.0 : 0.0);

        //2. Hidden layer gradients
        double[] dh = new double[h_size];
        for(int col = 0; col < h_size; col++)
        {
            double sum = 0.0;
            for(int j = 0; j < v_size; j++)
                sum = sum + (W2[col][j] * g[j]);
            dh[col] = sum;
        }

        double[] dz = new double[h_size];
        for(int col = 0; col < h_size; col++)
            dz[col] = dh[col] * (1.0 - (fc.h[col] * fc.h[col]));

        //3. Gradient arriving at net_in (size = 2*D)
        double[] d_net_in = new double[2 * d];
        for(int row = 0; row < 2 * d; row++)
        {
            double sum = 0.0;
            for(int col = 0; col < h_size; col++)
                sum = sum + (W1[row][col] * dz[col]);
            d_net_in[row] = sum;
        }

        double[] d_att_out = new double[d];
        double[] dx_res = new double[d];
        for(int m = 0; m < d; m++)
        {
            d_att_out[m] = d_net_in[m];
            dx_res[m] = d_net_in[d + m];
        }

        //4. Attention Backward
        //dv_i = a_i * dOut
        double[][] dv = new double[n][d];
        for(int i = 0; i < n; i++)
            for(int m = 0; m < d; m++)
                dv[i][m] = fc.a[i] * d_att_out[m];

        //da_i = dOut . v_i
        double[] da = new double[n];
        double sum_a_da = 0.0;
        for(int i = 0; i < n; i++)
        {
            double dot = 0.0;
            for(int m = 0; m < d; m++)
                dot = dot + (d_att_out[m] * fc.v[i][m]);
            da[i] = dot;
            sum_a_da = sum_a_da + (fc.a[i] * da[i]);
        }

        //ds_i = a_i * (da_i - sum_j a_j * da_j)
        double[] ds = new double[n];
        for(int i = 0; i < n; i++)
            ds[i] = fc.a[i] * (da[i] - sum_a_da);

        //Scale by 1/sqrt(D)
        //dk_i = (ds_i / scale) * q
        //dq = sum_i (ds_i / scale) * k_i
        double[][] dk = new double[n][d];
        double[] dq = new double[d];
        for(int i = 0; i < n; i++)
        {
            double s_scaled = ds[i] / scale;
            for(int m = 0; m < d; m++)
            {
                dk[i][m] = s_scaled * fc.q[m];
                dq[m] = dq[m] + (s_scaled * fc.k[i][m]);
            }
        }

        //Update W2 and b2
        for(int j = 0; j < v_size; j++)
        {
            b2[j] = b2[j] - (lr * g[j]);
            for(int col = 0; col < h_size; col++)
                W2[col][j] = W2[col][j] - (lr * fc.h[col] * g[j]);
        }

        //Update W1 and b1
        for(int col = 0; col < h_size; col++)
        {
            b1[col] = b1[col] - (lr * dz[col]);
            for(int row = 0; row < 2 * d; row++)
                W1[row][col] = W1[row][col] - (lr * fc.net_in[row] * dz[col]);
        }

        //Accumulate gradients for x_i
        double[][] dx = new double[n][d];

        //From v_i: x_i * Wv => dx_i += dv_i * Wv^T
        //From k_i: x_i * Wk => dx_i += dk_i * Wk^T
        for(int i = 0; i < n; i++)
        {
            for(int row = 0; row < d; row++)
            {
                double sum_v = 0.0;
                double sum_k = 0.0;
                for(int col = 0; col < d; col++)
                {
                    sum_v = sum_v + (dv[i][col] * Wv[row][col]);
                    sum_k = sum_k + (dk[i][col] * Wk[row][col]);
                }
                dx[i][row] = dx[i][row] + sum_v + sum_k;
            }
        }

        //From q: x[n-1] * Wq => dx[n-1] += dq * Wq^T
        //From residual: dx[n-1] += dx_res
        for(int row = 0; row < d; row++)
        {
            double sum_q = 0.0;
            for(int col = 0; col < d; col++)
                sum_q = sum_q + (dq[col] * Wq[row][col]);
            dx[n - 1][row] = dx[n - 1][row] + sum_q + dx_res[row];
        }

        //Update Wq: x[n-1]^T * dq
        for(int row = 0; row < d; row++)
            for(int col = 0; col < d; col++)
                Wq[row][col] = Wq[row][col] - (lr * fc.x[n - 1][row] * dq[col]);

        //Update Wk and Wv: sum_i x_i^T * dk_i and sum_i x_i^T * dv_i
        for(int row = 0; row < d; row++)
        {
            for(int col = 0; col < d; col++)
            {
                double sum_wk = 0.0;
                double sum_wv = 0.0;
                for(int i = 0; i < n; i++)
                {
                    sum_wk = sum_wk + (fc.x[i][row] * dk[i][col]);
                    sum_wv = sum_wv + (fc.x[i][row] * dv[i][col]);
                }
                Wk[row][col] = Wk[row][col] - (lr * sum_wk);
                Wv[row][col] = Wv[row][col] - (lr * sum_wv);
            }
        }

        //Update Embeddings C and Position vectors P
        for(int i = 0; i < n; i++)
        {
            int char_row = context[i];
            for(int m = 0; m < d; m++)
            {
                C[char_row][m] = C[char_row][m] - (lr * dx[i][m]);
                P[i][m] = P[i][m] - (lr * dx[i][m]);
            }
        }
    }

    //Gradient check on Wk[0][0]
    public static double[] debug_gradient_test_Wk(int[] context, int target_idx,
                                                  double[][] C, double[][] P,
                                                  double[][] Wq, double[][] Wk, double[][] Wv,
                                                  double[][] W1, double[] b1, double[][] W2, double[] b2,
                                                  int n, int d, int h_size, int v_size)
    {
        ForwardCache fc1 = forward(context, target_idx, C, P, Wq, Wk, Wv, W1, b1, W2, b2, n, d, h_size, v_size);
        double loss1 = fc1.loss;

        //Compute analytic gradient for Wk[0][0]
        double scale = Math.sqrt(d);
        double[] g = new double[v_size];
        for(int j = 0; j < v_size; j++)
            g[j] = fc1.probs[j] - ((j == target_idx) ? 1.0 : 0.0);

        double[] dh = new double[h_size];
        for(int col = 0; col < h_size; col++)
        {
            double sum = 0.0;
            for(int j = 0; j < v_size; j++)
                sum = sum + (W2[col][j] * g[j]);
            dh[col] = sum;
        }

        double[] dz = new double[h_size];
        for(int col = 0; col < h_size; col++)
            dz[col] = dh[col] * (1.0 - (fc1.h[col] * fc1.h[col]));

        double[] d_net_in = new double[2 * d];
        for(int row = 0; row < 2 * d; row++)
        {
            double sum = 0.0;
            for(int col = 0; col < h_size; col++)
                sum = sum + (W1[row][col] * dz[col]);
            d_net_in[row] = sum;
        }

        double[] d_att_out = new double[d];
        for(int m = 0; m < d; m++)
            d_att_out[m] = d_net_in[m];

        double[] da = new double[n];
        double sum_a_da = 0.0;
        for(int i = 0; i < n; i++)
        {
            double dot = 0.0;
            for(int m = 0; m < d; m++)
                dot = dot + (d_att_out[m] * fc1.v[i][m]);
            da[i] = dot;
            sum_a_da = sum_a_da + (fc1.a[i] * da[i]);
        }

        double[] ds = new double[n];
        for(int i = 0; i < n; i++)
            ds[i] = fc1.a[i] * (da[i] - sum_a_da);

        double[][] dk = new double[n][d];
        for(int i = 0; i < n; i++)
        {
            double s_scaled = ds[i] / scale;
            for(int m = 0; m < d; m++)
                dk[i][m] = s_scaled * fc1.q[m];
        }

        double analytic_grad_Wk00 = 0.0;
        for(int i = 0; i < n; i++)
            analytic_grad_Wk00 = analytic_grad_Wk00 + (fc1.x[i][0] * dk[i][0]);

        //Numerical gradient with eps = 0.0001
        double eps = 0.0001;
        Wk[0][0] = Wk[0][0] + eps;
        ForwardCache fc2 = forward(context, target_idx, C, P, Wq, Wk, Wv, W1, b1, W2, b2, n, d, h_size, v_size);
        double loss2 = fc2.loss;
        double numerical_grad_Wk00 = (loss2 - loss1) / eps;
        Wk[0][0] = Wk[0][0] - eps;

        return new double[]{analytic_grad_Wk00, numerical_grad_Wk00};
    }

    //Train one epoch
    public static double train_one_epoch(double[][] C, double[][] P,
                                         double[][] Wq, double[][] Wk, double[][] Wv,
                                         double[][] W1, double[] b1, double[][] W2, double[] b2,
                                         String text, char[] array, double lr, int n, int d, int h_size, int v_size)
    {
        double total_loss = 0.0;
        int total_examples = text.length() - n;
        int[] context = new int[n];

        for(int i = 0; i < total_examples; i++)
        {
            for(int k = 0; k < n; k++)
                context[k] = find_ch_number(text.charAt(i + k), array);
            int target_idx = find_ch_number(text.charAt(i + n), array);

            ForwardCache fc = forward(context, target_idx, C, P, Wq, Wk, Wv, W1, b1, W2, b2, n, d, h_size, v_size);
            total_loss = total_loss + fc.loss;

            backward_and_nudge(fc, context, target_idx, C, P, Wq, Wk, Wv, W1, b1, W2, b2, lr, n, d, h_size, v_size);
        }

        return total_loss / total_examples;
    }

    //Evaluate validation loss (forward pass only, no nudges)
    public static double compute_loss_on_text(double[][] C, double[][] P,
                                              double[][] Wq, double[][] Wk, double[][] Wv,
                                              double[][] W1, double[] b1, double[][] W2, double[] b2,
                                              String text, char[] array, int n, int d, int h_size, int v_size)
    {
        double total_loss = 0.0;
        int total_examples = text.length() - n;
        int[] context = new int[n];

        for(int i = 0; i < total_examples; i++)
        {
            for(int k = 0; k < n; k++)
                context[k] = find_ch_number(text.charAt(i + k), array);
            int target_idx = find_ch_number(text.charAt(i + n), array);

            ForwardCache fc = forward(context, target_idx, C, P, Wq, Wk, Wv, W1, b1, W2, b2, n, d, h_size, v_size);
            total_loss = total_loss + fc.loss;
        }

        return total_loss / total_examples;
    }
}
