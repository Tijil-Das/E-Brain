public class Driver
{
    static String text = Data.chunk1;
    static char[] char_present_in_text;
    static int v;
    static int d = 10;
    static int h_size = 100;

    public static void main(String[] args)
    {
        Library.fill_the_characters_present_in_text(text);
        char_present_in_text = Library.ch_present;
        v = char_present_in_text.length;

        System.out.println("===================================================================");
        System.out.println("                    STAGE 7: ATTENTION MECHANISM                   ");
        System.out.println("===================================================================\n");

        //1. Hand Check
        System.out.println("--- 1. HAND CHECK TEST (D=2, N=3, skip sqrt(D)) ---");
        double[][] hand = Library.run_hand_check();
        System.out.printf("Scores:  [%.1f, %.1f, %.1f]%n", hand[0][0], hand[0][1], hand[0][2]);
        System.out.printf("Softmax: [%.3f, %.3f, %.3f]  (Expected: ~0.422, ~0.155, ~0.422)%n", hand[1][0], hand[1][1], hand[1][2]);
        System.out.printf("Output:  [%.3f, %.3f]         (Expected: ~[1.266, 1.042])%n%n", hand[2][0], hand[2][1]);

        //Split text 90% train, 10% validation
        int split_idx = (int)(text.length() * 0.90);
        String train_set = text.substring(0, split_idx);
        String val_set = text.substring(split_idx);

        //2. Test Attention with N = 4 (Baseline comparison)
        System.out.println("--- 2. ATTENTION MODEL (N = 4, D = 10, H = 100) ---");
        run_attention_run(4, train_set, val_set, 10, 0.01);

        //3. Scale Attention to N = 16 (Cheap context scaling!)
        System.out.println("\n--- 3. SCALING ATTENTION TO N = 16 (Longer Context) ---");
        run_attention_run(16, train_set, val_set, 10, 0.01);
    }

    public static void run_attention_run(int n, String train_set, String val_set, int epochs, double lr)
    {
        //Initialize weights
        double[][] C = Library.make_random_matrix(v, d, 0.1);
        double[][] P = Library.make_random_matrix(n, d, 0.1);
        double[][] Wq = Library.make_random_matrix(d, d, 0.1);
        double[][] Wk = Library.make_random_matrix(d, d, 0.1);
        double[][] Wv = Library.make_random_matrix(d, d, 0.1);
        double[][] W1 = Library.make_random_matrix(2 * d, h_size, 0.05);
        double[] b1 = Library.make_zeros_vector(h_size);
        double[][] W2 = Library.make_random_matrix(h_size, v, 0.05);
        double[] b2 = Library.make_zeros_vector(v);

        //Gradient Check on Wk[0][0]
        int[] init_context = new int[n];
        for(int k = 0; k < n; k++)
            init_context[k] = Library.find_ch_number(train_set.charAt(k), char_present_in_text);
        int init_target = Library.find_ch_number(train_set.charAt(n), char_present_in_text);

        double[] grad_check = Library.debug_gradient_test_Wk(init_context, init_target, C, P, Wq, Wk, Wv, W1, b1, W2, b2, n, d, h_size, v);
        System.out.printf("Gradient Check (Wk[0][0], N=%d) -> Analytic: %.8f | Numerical: %.8f (Match: %b)%n",
            n, grad_check[0], grad_check[1], Math.abs(grad_check[0] - grad_check[1]) < 1e-5);

        double initial_val = Library.compute_loss_on_text(C, P, Wq, Wk, Wv, W1, b1, W2, b2, val_set, char_present_in_text, n, d, h_size, v);
        System.out.printf("Initial Validation Loss: %.4f%n", initial_val);

        double best_val = Double.MAX_VALUE;
        int best_ep = 1;
        double final_train = 0.0;
        double final_val = 0.0;

        for(int ep = 1; ep <= epochs; ep++)
        {
            double tr = Library.train_one_epoch(C, P, Wq, Wk, Wv, W1, b1, W2, b2, train_set, char_present_in_text, lr, n, d, h_size, v);
            double val = Library.compute_loss_on_text(C, P, Wq, Wk, Wv, W1, b1, W2, b2, val_set, char_present_in_text, n, d, h_size, v);

            if(val < best_val)
            {
                best_val = val;
                best_ep = ep;
            }

            final_train = tr;
            final_val = val;

            System.out.printf("  Epoch %2d / %2d - Train Loss: %.4f | Val Loss: %.4f | Gap: %+.4f%n",
                ep, epochs, tr, val, (val - tr));
        }

        System.out.printf("[N = %d Results] Final Train Loss: %.4f | Final Val Loss: %.4f | Best Val Loss: %.4f (Epoch %d)%n",
            n, final_train, final_val, best_val, best_ep);
    }
}
