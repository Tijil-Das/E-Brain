import java.util.concurrent.ThreadLocalRandom;

public class ChangeA
{
    static String text = Data.chunk1;
    static char[] char_present_in_text;
    static int v;
    static int n = 4; // 4 characters of context
    static int d = 10;
    static int input_dim = n * d; // 40 numbers
    static double[][] C;
    static double[][] W;
    static double[] b;
    static double learning_rate = 0.01;
    static int epochs = 20;

    public static void main(String[] args)
    {
        fill_the_characters_present_in_text(text);
        v = char_present_in_text.length;

        System.out.println("=== Stage 4 - Change A: Longer Memory (N = 4 characters, No Hidden Layer) ===");
        System.out.println("Vocabulary size V = " + v + ", N = " + n + ", D = " + d + ", Input dimension = " + input_dim);
        System.out.printf("Theoretical initial loss ln(V) = ln(%d) = %.4f%n%n", v, Math.log(v));

        C = make_embedding_table(v, d);
        W = make_weight_matrix(input_dim, v);
        b = make_bias_vector(v);

        double starting_loss = compute_loss_on_text();
        System.out.printf("Starting loss before training: %.4f%n%n", starting_loss);

        System.out.println("--- Starting Training (" + epochs + " epochs, learning rate = " + learning_rate + ") ---");
        double final_loss = starting_loss;
        for(int epoch = 1; epoch <= epochs; epoch++)
        {
            double epoch_loss = train_one_epoch();
            final_loss = epoch_loss;
            System.out.printf("Epoch %2d / %2d - Average Loss: %.4f%n", epoch, epochs, epoch_loss);
        }

        System.out.println();
        System.out.printf("Change A Complete! Starting loss: %.4f | Final loss: %.4f%n%n", starting_loss, final_loss);
    }

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
        char_present_in_text = new char[c];
        for(int i = 0; i < c; i++)
        {
            char_present_in_text[i] = temp[i];
        }
    }

    public static int find_ch_number(char ch)
    {
        for(int i = 0; i < char_present_in_text.length; i++)
        {
            if(char_present_in_text[i] == ch)
                return i;
        }
        return -1;
    }

    public static double[][] make_embedding_table(int v, int d)
    {
        double[][] C = new double[v][d];
        for(int i = 0; i < v; i++)
            for(int j = 0; j < d; j++)
                C[i][j] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        return C;
    }

    public static double[][] make_weight_matrix(int in_dim, int v)
    {
        double[][] W = new double[in_dim][v];
        for(int i = 0; i < in_dim; i++)
            for(int j = 0; j < v; j++)
                W[i][j] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        return W;
    }

    public static double[] make_bias_vector(int v)
    {
        return new double[v];
    }

    public static double[] build_input(int[] context_indices)
    {
        double[] input = new double[input_dim];
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

    public static double[] compute_scores(double[] input)
    {
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

    public static double[] calculate_softmax(double[] scores)
    {
        double max_score = scores[0];
        for(int i = 1; i < v; i++)
            if(scores[i] > max_score)
                max_score = scores[i];

        double[] exp_scores = new double[v];
        double sum = 0.0;
        for(int i = 0; i < v; i++)
        {
            exp_scores[i] = Math.exp(scores[i] - max_score);
            sum = sum + exp_scores[i];
        }

        double[] probs = new double[v];
        for(int i = 0; i < v; i++)
            probs[i] = exp_scores[i] / sum;
        return probs;
    }

    public static double train_one_epoch()
    {
        double total_loss = 0.0;
        int total_examples = text.length() - n;
        int[] context = new int[n];

        for(int i = 0; i < total_examples; i++)
        {
            for(int k = 0; k < n; k++)
            {
                context[k] = find_ch_number(text.charAt(i + k));
            }
            int target_idx = find_ch_number(text.charAt(i + n));

            double[] input = build_input(context);
            double[] scores = compute_scores(input);
            double[] probs = calculate_softmax(scores);

            total_loss = total_loss + (-Math.log(probs[target_idx]));

            // Backward pass
            double[] g = new double[v];
            for(int j = 0; j < v; j++)
            {
                double target = (j == target_idx) ? 1.0 : 0.0;
                g[j] = probs[j] - target;
            }

            // Input gradient
            double[] grad_input = new double[input_dim];
            for(int k = 0; k < input_dim; k++)
            {
                double sum = 0.0;
                for(int j = 0; j < v; j++)
                    sum = sum + (W[k][j] * g[j]);
                grad_input[k] = sum;
            }

            // Update b
            for(int j = 0; j < v; j++)
                b[j] = b[j] - (learning_rate * g[j]);

            // Update W
            for(int k = 0; k < input_dim; k++)
                for(int j = 0; j < v; j++)
                    W[k][j] = W[k][j] - (learning_rate * input[k] * g[j]);

            // Split gradients back into embedding rows
            for(int idx = 0; idx < n; idx++)
            {
                int char_row = context[idx];
                for(int k = 0; k < d; k++)
                {
                    C[char_row][k] = C[char_row][k] - (learning_rate * grad_input[idx * d + k]);
                }
            }
        }

        return total_loss / total_examples;
    }

    public static double compute_loss_on_text()
    {
        double total_loss = 0.0;
        int total_examples = text.length() - n;
        int[] context = new int[n];

        for(int i = 0; i < total_examples; i++)
        {
            for(int k = 0; k < n; k++)
                context[k] = find_ch_number(text.charAt(i + k));
            int target_idx = find_ch_number(text.charAt(i + n));

            double[] input = build_input(context);
            double[] scores = compute_scores(input);
            double[] probs = calculate_softmax(scores);

            total_loss = total_loss + (-Math.log(probs[target_idx]));
        }

        return total_loss / total_examples;
    }
}
