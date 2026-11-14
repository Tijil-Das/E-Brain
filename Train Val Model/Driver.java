import java.util.Scanner;

public class Driver
{
    static Scanner sc = new Scanner(System.in); 
    static String text = Data.chunk1;
    static char[] char_present_in_text;
    static int v;
    static int n = 4; // Context length (4 characters)
    static int d = 10; // Embedding dimension
    static int h = 100; // Hidden layer size
    static int input_dim = n * d; // 40 numbers
    static double[][] C;
    static double[][] W1;
    static double[] b1;
    static double[][] W2;
    static double[] b2;
    static double learning_rate = 0.01;
    static int epochs = 20;

    public static void main(String[] args)
    {
        main();
    }

    public static void main()
    {
        Library.fill_the_characters_present_in_text(text);
        char_present_in_text = Library.ch_present;
        v = char_present_in_text.length;

        //Step 1: Split text into 90% training and 10% validation
        int split_index = (int)(text.length() * 0.90);
        String train_text = text.substring(0, split_index);
        String val_text = text.substring(split_index);

        System.out.println("=== Stage 5: Train vs Validation Loss (Overfitting Check) ===");
        System.out.printf("Total characters: %d | Train set: %d (90%%) | Val set: %d (10%%)%n", 
            text.length(), train_text.length(), val_text.length());
        System.out.println("Context N = " + n + ", Embedding D = " + d + ", Hidden H = " + h);
        System.out.printf("Theoretical initial loss ln(V) = ln(%d) = %.4f%n%n", v, Math.log(v));

        //Initialize parameters
        C = Library.make_embedding_table(v, d);
        W1 = Library.make_W1_matrix(input_dim, h);
        b1 = Library.make_b1_vector(h);
        W2 = Library.make_W2_matrix(h, v);
        b2 = Library.make_b2_vector(v);

        //Initial losses before any training
        double initial_train_loss = Library.compute_loss_on_text(C, W1, b1, W2, b2, train_text, char_present_in_text, n, d);
        double initial_val_loss = Library.compute_loss_on_text(C, W1, b1, W2, b2, val_text, char_present_in_text, n, d);
        System.out.printf("Initial Losses -> Train: %.4f | Val: %.4f | Gap: %.4f%n%n", 
            initial_train_loss, initial_val_loss, (initial_val_loss - initial_train_loss));

        //Step 2 & 3: Training loop and measure both losses
        System.out.println("--- Starting Training (" + epochs + " epochs, learning rate = " + learning_rate + ") ---");
        double best_val_loss = Double.MAX_VALUE;
        int best_epoch = 1;

        double final_train_loss = initial_train_loss;
        double final_val_loss = initial_val_loss;

        for(int epoch = 1; epoch <= epochs; epoch++)
        {
            //Train on 90% training text only
            double train_loss = Library.train_one_epoch(C, W1, b1, W2, b2, train_text, char_present_in_text, learning_rate, n, d);
            
            //Measure loss on 10% held-out validation text (forward pass only, no nudges)
            double val_loss = Library.compute_loss_on_text(C, W1, b1, W2, b2, val_text, char_present_in_text, n, d);

            double gap = val_loss - train_loss;

            if(val_loss < best_val_loss)
            {
                best_val_loss = val_loss;
                best_epoch = epoch;
            }

            final_train_loss = train_loss;
            final_val_loss = val_loss;

            System.out.printf("Epoch %2d / %2d - Train Loss: %.4f | Val Loss: %.4f | Gap: %+.4f%n", 
                epoch, epochs, train_loss, val_loss, gap);
        }

        System.out.println();
        System.out.printf("Training complete! Final Train Loss: %.4f | Final Val Loss: %.4f | Gap: %.4f%n", 
            final_train_loss, final_val_loss, (final_val_loss - final_train_loss));
        System.out.printf("Best Validation Loss occurred at Epoch %d (Val Loss: %.4f)%n%n", best_epoch, best_val_loss);

        //Analyze drift
        double initial_gap = initial_val_loss - initial_train_loss;
        double final_gap = final_val_loss - final_train_loss;
        System.out.println("--- Overfitting Analysis ---");
        if(final_gap > initial_gap + 0.15 || final_val_loss > best_val_loss + 0.05)
        {
            System.out.println("Status: Overfitting observed! The gap between train and validation is widening.");
        }
        else
        {
            System.out.printf("Status: Healthy generalization! Train and Val losses fell closely together (gap is %.4f).%n", final_gap);
        }
        System.out.println();

        //Sample generation
        System.out.println("Enter how many characters to generate:");
        long num_chars = 300;
        if(sc.hasNextLong())
        {
            num_chars = num_chars = sc.nextLong();
        }
        else if(sc.hasNextInt())
        {
            num_chars = sc.nextInt();
        }

        System.out.println("\n--- Generating " + num_chars + " characters ---");
        int[] current_context = new int[n];
        for(int k = 0; k < n; k++)
        {
            current_context[k] = (int)(Math.random() * v);
            System.out.print(char_present_in_text[current_context[k]]);
        }

        for(long i = 1; i <= num_chars - n; i++)
        {
            char next_char = Library.sample_next(current_context, C, W1, b1, W2, b2, d);
            System.out.print(next_char);

            for(int k = 0; k < n - 1; k++)
            {
                current_context[k] = current_context[k + 1];
            }
            current_context[n - 1] = Library.find_ch_number(next_char, char_present_in_text);
        }
        System.out.println("\n");
    }
}
