import java.util.Scanner;
public class Driver
{
    static Scanner sc=new Scanner(System.in); 
    static String text=Data.chunk1;
    static char[] char_present_in_text=Library.ch_present;
    static long[][] matrix;
    static int array_length;
    public static void main()
    {
        Library.fill_the_characters_present_in_text(text);
        array_length=char_present_in_text.length;
        matrix=Library.make_matrix_and_fill_values(char_present_in_text, text);
        System.out.println("Enter how many characters to generate");
        long n=sc.nextInt();
        char ch=Library.find_ch_at_a_number((int)Library.pick_random(array_length+1), char_present_in_text);
        char generated='\0';
        for(long i=1; i<=n; i++)
        {
            if(i==1)
                generated=Library.generate(ch, matrix);
            else
                generated=Library.generate(generated, matrix);
                
            System.out.print(generated);
        }
    }
}