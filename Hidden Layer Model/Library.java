import java.util.concurrent.ThreadLocalRandom;

public class Library 
{
    static char[] ch_present;

    public static void fill_the_characters_present_in_text(String data)
    {
        char[] temp = new char[256];
        int c = 0;
        String text = data;
        for(int i = 0; i < text.length(); i++)
        {
            char ch = text.charAt(i);
            if(check_if_char_already_filled(ch, temp, c) == false)
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

    public static boolean check_if_char_already_filled(char ch, char[] array, int length)
    {
        for(int i = 0; i < length; i++)
        {
            if(array[i] == ch)
                return true;
        }
        return false;
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

    //Embedding table V x D
    public static double[][] make_embedding_table(int v, int d)
    {
        double[][] C = new double[v][d];
        for(int i = 0; i < v; i++)
        {
            for(int j = 0; j < d; j++)
            {
                C[i][j] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
            }
        }
        return C;
    }

    //Hidden table W1: (N*D) rows x H columns, scaled small
    public static double[][] make_W1_matrix(int in_dim, int h)
    {
        double[][] W1 = new double[in_dim][h];
        for(int i = 0; i < in_dim; i++)
        {
            for(int j = 0; j < h; j++)
            {
                W1[i][j] = ThreadLocalRandom.current().nextDouble(-0.05, 0.05);
            }
        }
        return W1;
    }

    //Bias list b1: H zeros
    public static double[] make_b1_vector(int h)
    {
        return new double[h];
    }

    //Output table W2: H rows x V columns, scaled small
    public static double[][] make_W2_matrix(int h, int v)
    {
        double[][] W2 = new double[h][v];
        for(int i = 0; i < h; i++)
        {
            for(int j = 0; j < v; j++)
            {
                W2[i][j] = ThreadLocalRandom.current().nextDouble(-0.05, 0.05);
            }
        }
        return W2;
    }

    //Bias list b2: V zeros
    public static double[] make_b2_vector(int v)
    {
        return new double[v];
    }

    //Build input vector of length N*D by concatenating character embeddings
    public static double[] build_input(int[] context_indices, double[][] C, int d)
    {
        int n = context_indices.length;
        double[] input = new double[n * d];
        for(int idx = 0; idx < n; idx++)
        {
            int char_row = context_indices[idx];
            for(int k = 0; k < d; k++)
            {
                input[idx * d + k] = C[char_row][k];
            }
        }
        return input;
    }

    //Compute hidden activations h_k = tanh(sum of input_i * W1[i][k] + b1_k)
    public static double[] compute_hidden(double[] input, double[][] W1, double[] b1)
    {
        int h = b1.length;
        int in_dim = input.length;
        double[] hidden = new double[h];

        for(int k = 0; k < h; k++)
        {
            double sum = 0.0;
            for(int i = 0; i < in_dim; i++)
            {
                sum = sum + (input[i] * W1[i][k]);
            }
            double z = sum + b1[k];
            hidden[k] = Math.tanh(z);
        }
        return hidden;
    }

    //Compute output scores = hidden * W2 + b2
    public static double[] compute_scores(double[] hidden, double[][] W2, double[] b2)
    {
        int v = b2.length;
        int h = hidden.length;
        double[] scores = new double[v];

        for(int j = 0; j < v; j++)
        {
            double sum = 0.0;
            for(int k = 0; k < h; k++)
            {
                sum = sum + (hidden[k] * W2[k][j]);
            }
            scores[j] = sum + b2[j];
        }
        return scores;
    }

    //Softmax
    public static double[] calculate_softmax(double[] scores)
    {
        int v = scores.length;
        double max_score = scores[0];
        for(int i = 1; i < v; i++)
        {
            if(scores[i] > max_score)
                max_score = scores[i];
        }

        double[] exp_scores = new double[v];
        double total_exp_sum = 0.0;
        for(int i = 0; i < v; i++)
        {
            exp_scores[i] = Math.exp(scores[i] - max_score);
            total_exp_sum = total_exp_sum + exp_scores[i];
        }

        double[] probs = new double[v];
        for(int i = 0; i < v; i++)
        {
            probs[i] = exp_scores[i] / total_exp_sum;
        }
        return probs;
    }

    //Backward pass and nudge
    public static void backward_and_nudge(double[] input, double[] hidden, double[] probs, int[] context_indices, int target_idx,
                                          double[][] C, double[][] W1, double[] b1, double[][] W2, double[] b2,
                                          double learning_rate, int d)
    {
        int v = b2.length;
        int h = hidden.length;
        int in_dim = input.length;
        int n = context_indices.length;

        //g = probabilities - one-hot
        double[] g = new double[v];
        for(int j = 0; j < v; j++)
        {
            double target = (j == target_idx) ? 1.0 : 0.0;
            g[j] = probs[j] - target;
        }

        //Hidden gradient: dh_k = sum over j of W2[k][j] * g_j
        double[] dh = new double[h];
        for(int k = 0; k < h; k++)
        {
            double sum = 0.0;
            for(int j = 0; j < v; j++)
            {
                sum = sum + (W2[k][j] * g[j]);
            }
            dh[k] = sum;
        }

        //Pass through tanh: dz_k = dh_k * (1 - h_k^2)
        double[] dz = new double[h];
        for(int k = 0; k < h; k++)
        {
            dz[k] = dh[k] * (1.0 - (hidden[k] * hidden[k]));
        }

        //Input gradient: dx_i = sum over k of W1[i][k] * dz_k
        double[] dx = new double[in_dim];
        for(int i = 0; i < in_dim; i++)
        {
            double sum = 0.0;
            for(int k = 0; k < h; k++)
            {
                sum = sum + (W1[i][k] * dz[k]);
            }
            dx[i] = sum;
        }

        //Update W2 and b2:
        for(int j = 0; j < v; j++)
        {
            b2[j] = b2[j] - (learning_rate * g[j]);
            for(int k = 0; k < h; k++)
            {
                double grad_w2 = hidden[k] * g[j];
                W2[k][j] = W2[k][j] - (learning_rate * grad_w2);
            }
        }

        //Update W1 and b1:
        for(int k = 0; k < h; k++)
        {
            b1[k] = b1[k] - (learning_rate * dz[k]);
            for(int i = 0; i < in_dim; i++)
            {
                double grad_w1 = input[i] * dz[k];
                W1[i][k] = W1[i][k] - (learning_rate * grad_w1);
            }
        }

        //Split input gradients back into embedding rows
        for(int idx = 0; idx < n; idx++)
        {
            int char_row = context_indices[idx];
            for(int k = 0; k < d; k++)
            {
                C[char_row][k] = C[char_row][k] - (learning_rate * dx[idx * d + k]);
            }
        }
    }

    //Gradient check on W1[0][0]
    public static double[] debug_gradient_test_W1(int[] context, int target_idx, double[][] C, double[][] W1, double[] b1, double[][] W2, double[] b2, int d)
    {
        double[] input = build_input(context, C, d);
        double[] hidden = compute_hidden(input, W1, b1);
        double[] scores = compute_scores(hidden, W2, b2);
        double[] probs = calculate_softmax(scores);
        double loss1 = -Math.log(probs[target_idx]);

        //Analytical gradient
        int v = b2.length;
        int h = hidden.length;
        double target = (0 == target_idx) ? 1.0 : 0.0;
        double[] g = new double[v];
        for(int j = 0; j < v; j++)
            g[j] = probs[j] - ((j == target_idx) ? 1.0 : 0.0);

        double dh0 = 0.0;
        for(int j = 0; j < v; j++)
            dh0 = dh0 + (W2[0][j] * g[j]);
        double dz0 = dh0 * (1.0 - (hidden[0] * hidden[0]));
        double analytic_grad = input[0] * dz0;

        //Numerical gradient with eps = 0.0001
        double eps = 0.0001;
        W1[0][0] = W1[0][0] + eps;
        double[] hidden2 = compute_hidden(input, W1, b1);
        double[] scores2 = compute_scores(hidden2, W2, b2);
        double[] probs2 = calculate_softmax(scores2);
        double loss2 = -Math.log(probs2[target_idx]);
        double numerical_grad = (loss2 - loss1) / eps;
        W1[0][0] = W1[0][0] - eps;

        return new double[]{analytic_grad, numerical_grad};
    }

    //Train one epoch
    public static double train_one_epoch(double[][] C, double[][] W1, double[] b1, double[][] W2, double[] b2,
                                         String text, char[] array, double learning_rate, int n, int d)
    {
        double total_loss = 0.0;
        int total_examples = text.length() - n;
        int[] context = new int[n];

        for(int i = 0; i < total_examples; i++)
        {
            for(int k = 0; k < n; k++)
            {
                context[k] = find_ch_number(text.charAt(i + k), array);
            }
            int target_idx = find_ch_number(text.charAt(i + n), array);

            double[] input = build_input(context, C, d);
            double[] hidden = compute_hidden(input, W1, b1);
            double[] scores = compute_scores(hidden, W2, b2);
            double[] probs = calculate_softmax(scores);

            total_loss = total_loss + (-Math.log(probs[target_idx]));

            backward_and_nudge(input, hidden, probs, context, target_idx, C, W1, b1, W2, b2, learning_rate, d);
        }

        return total_loss / total_examples;
    }

    //Compute loss on entire text without updating
    public static double compute_loss_on_text(double[][] C, double[][] W1, double[] b1, double[][] W2, double[] b2,
                                              String text, char[] array, int n, int d)
    {
        double total_loss = 0.0;
        int total_examples = text.length() - n;
        int[] context = new int[n];

        for(int i = 0; i < total_examples; i++)
        {
            for(int k = 0; k < n; k++)
                context[k] = find_ch_number(text.charAt(i + k), array);
            int target_idx = find_ch_number(text.charAt(i + n), array);

            double[] input = build_input(context, C, d);
            double[] hidden = compute_hidden(input, W1, b1);
            double[] scores = compute_scores(hidden, W2, b2);
            double[] probs = calculate_softmax(scores);

            total_loss = total_loss + (-Math.log(probs[target_idx]));
        }

        return total_loss / total_examples;
    }

    //Sample next character given context
    public static char sample_next(int[] context, double[][] C, double[][] W1, double[] b1, double[][] W2, double[] b2, int d)
    {
        double[] input = build_input(context, C, d);
        double[] hidden = compute_hidden(input, W1, b1);
        double[] scores = compute_scores(hidden, W2, b2);
        double[] probs = calculate_softmax(scores);

        double random = ThreadLocalRandom.current().nextDouble();
        int i = 0;
        double sum = 0.0;
        for(i = 0; i < ch_present.length; i++)
        {
            sum = sum + probs[i];
            if(sum > random)
                break;
        }

        if(i >= ch_present.length)
            i = ch_present.length - 1;

        return find_ch_at_a_number(i, ch_present);
    }
}
