import java.util.concurrent.ThreadLocalRandom;

public class Library 
{
    //Array holding all unique characters present in the text
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
        boolean flag = false;
        for(int i = 0; i < length; i++)
        {
            if(array[i] == ch)
            {
                flag = true;
                break;
            }
        }
        return flag;
    }

    public static int find_ch_number(char ch, char[] array)//find the zero-based position of a character in an array. if character not present, return -1 
    {
        for(int i = 0; i < array.length; i++)
        {
            if(array[i] == ch)
                return i;
        }
        return -1;
    }

    public static Character find_ch_at_a_number(int n, char[] array)//inputs the array and the zero-based position number and outputs the character in that array at that position
    {
        if(n < 0 || n >= array.length)
            return null;
        return array[n];
    }

    //Step 1: Embedding table (the vectors) of size V x D filled with small random numbers
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

    //Step 3: Weight table W of size (2*D) x V filled with small random numbers
    public static double[][] make_weight_matrix(int input_dim, int v)
    {
        double[][] W = new double[input_dim][v];
        for(int i = 0; i < input_dim; i++)
        {
            for(int j = 0; j < v; j++)
            {
                W[i][j] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
            }
        }
        return W;
    }

    //Step 3: Bias list b of V numbers initialized to zero
    public static double[] make_bias_vector(int v)
    {
        double[] b = new double[v];
        for(int i = 0; i < v; i++)
        {
            b[i] = 0.0;
        }
        return b;
    }

    //Step 2: Build the input by placing vectors of older and newer characters side by side
    public static double[] build_input(int c1_index, int c2_index, double[][] C, int d)
    {
        double[] input = new double[2 * d];
        //Fetch older character vector
        for(int i = 0; i < d; i++)
        {
            input[i] = C[c1_index][i];
        }
        //Fetch newer character vector
        for(int i = 0; i < d; i++)
        {
            input[d + i] = C[c2_index][i];
        }
        return input;
    }

    //Step 3: Turn the input into V scores: score_j = (sum over 20 of input_i * W[i][j]) + b_j
    public static double[] compute_scores(double[] input, double[][] W, double[] b)
    {
        int v = b.length;
        int input_dim = input.length;
        double[] scores = new double[v];

        for(int j = 0; j < v; j++)
        {
            double sum = 0.0;
            for(int i = 0; i < input_dim; i++)
            {
                sum = sum + (input[i] * W[i][j]);
            }
            scores[j] = sum + b[j];
        }
        return scores;
    }

    //Softmax function
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

        double[] probabilities = new double[v];
        for(int i = 0; i < v; i++)
        {
            probabilities[i] = exp_scores[i] / total_exp_sum;
        }
        return probabilities;
    }

    //Step 4: Nudge parameters
    public static void nudge(double[] input, double[] probabilities, int c1_row, int c2_row, int c3_col, double[][] C, double[][] W, double[] b, double learning_rate, int d)
    {
        int v = b.length;
        int input_dim = input.length;

        //List g of V numbers: probability - (1 if correct else 0)
        double[] g = new double[v];
        for(int j = 0; j < v; j++)
        {
            double target = 0.0;
            if(j == c3_col)
                target = 1.0;
            g[j] = probabilities[j] - target;
        }

        //input_i gradient: sum over j of W[i][j] * g_j (computed before updating W)
        double[] grad_input = new double[input_dim];
        for(int i = 0; i < input_dim; i++)
        {
            double sum = 0.0;
            for(int j = 0; j < v; j++)
            {
                sum = sum + (W[i][j] * g[j]);
            }
            grad_input[i] = sum;
        }

        //Update b_j: gradient is g_j
        for(int j = 0; j < v; j++)
        {
            b[j] = b[j] - (learning_rate * g[j]);
        }

        //Update W[i][j]: gradient is input_i * g_j
        for(int i = 0; i < input_dim; i++)
        {
            for(int j = 0; j < v; j++)
            {
                double grad_w = input[i] * g[j];
                W[i][j] = W[i][j] - (learning_rate * grad_w);
            }
        }

        //Split input gradients back: first d to older char, last d to newer char
        for(int k = 0; k < d; k++)
        {
            C[c1_row][k] = C[c1_row][k] - (learning_rate * grad_input[k]);
            C[c2_row][k] = C[c2_row][k] - (learning_rate * grad_input[d + k]);
        }
    }

    //Debug test for gradient verification
    public static void debug_gradient_test(double[][] C, double[][] W, double[] b, int c1_idx, int c2_idx, int c3_idx, int d)
    {
        double[] input = build_input(c1_idx, c2_idx, C, d);
        double[] scores = compute_scores(input, W, b);
        double[] probs = calculate_softmax(scores);
        double loss1 = -Math.log(probs[c3_idx]);

        //Analytical gradient for W[0][0] = input_0 * g_0
        double target0 = (c3_idx == 0) ? 1.0 : 0.0;
        double g0 = probs[0] - target0;
        double analytic_grad = input[0] * g0;

        //Numerical gradient with eps = 0.0001
        double eps = 0.0001;
        W[0][0] = W[0][0] + eps;
        double[] scores2 = compute_scores(input, W, b);
        double[] probs2 = calculate_softmax(scores2);
        double loss2 = -Math.log(probs2[c3_idx]);
        double numerical_grad = (loss2 - loss1) / eps;

        //Restore W[0][0]
        W[0][0] = W[0][0] - eps;

        System.out.printf("Debug Test -> Analytic gradient: %.6f | Numerical gradient: %.6f (Match: %b)%n%n", 
            analytic_grad, numerical_grad, Math.abs(analytic_grad - numerical_grad) < 1e-4);
    }

    //Train for one epoch over the text
    public static double train_one_epoch(double[][] C, double[][] W, double[] b, String text, char[] array, double learning_rate, int d)
    {
        double total_loss = 0.0;
        int total_triplets = text.length() - 2;

        for(int i = 0; i < total_triplets; i++)
        {
            char older_ch = text.charAt(i);
            char newer_ch = text.charAt(i + 1);
            char next_ch = text.charAt(i + 2);

            int c1_idx = find_ch_number(older_ch, array);
            int c2_idx = find_ch_number(newer_ch, array);
            int c3_idx = find_ch_number(next_ch, array);

            //Build 20-dim input
            double[] input = build_input(c1_idx, c2_idx, C, d);

            //Compute scores and softmax probabilities
            double[] scores = compute_scores(input, W, b);
            double[] probs = calculate_softmax(scores);

            //Loss
            double p = probs[c3_idx];
            total_loss = total_loss + (-Math.log(p));

            //Nudge parameters
            nudge(input, probs, c1_idx, c2_idx, c3_idx, C, W, b, learning_rate, d);
        }

        return total_loss / total_triplets;
    }

    //Compute average loss on text without updating parameters
    public static double compute_loss_on_text(double[][] C, double[][] W, double[] b, String text, char[] array, int d)
    {
        double total_loss = 0.0;
        int total_triplets = text.length() - 2;

        for(int i = 0; i < total_triplets; i++)
        {
            char older_ch = text.charAt(i);
            char newer_ch = text.charAt(i + 1);
            char next_ch = text.charAt(i + 2);

            int c1_idx = find_ch_number(older_ch, array);
            int c2_idx = find_ch_number(newer_ch, array);
            int c3_idx = find_ch_number(next_ch, array);

            double[] input = build_input(c1_idx, c2_idx, C, d);
            double[] scores = compute_scores(input, W, b);
            double[] probs = calculate_softmax(scores);

            total_loss = total_loss + (-Math.log(probs[c3_idx]));
        }

        return total_loss / total_triplets;
    }

    //Generate text by feeding the last two characters into the network
    public static char sample_next(int c1_idx, int c2_idx, double[][] C, double[][] W, double[] b, int d)
    {
        double[] input = build_input(c1_idx, c2_idx, C, d);
        double[] scores = compute_scores(input, W, b);
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
