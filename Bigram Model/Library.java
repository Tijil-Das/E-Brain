import java.util.concurrent.ThreadLocalRandom;
public class Library 
{
    //We need to declare an array whose length is atleast 94 because, in it we need to hold those characters that have already been checked
    static char[] ch_present=new char[94];

    public static void fill_the_characters_present_in_text(String data)
    {
        int c=0;
        String text = data;
        for(int i=0; i<text.length(); i++)//Iterate through each letter and fill the ch_present array
        {
            if(find_how_many_array_cells_filled()==94)
                break;
            char ch=text.charAt(i);
            if(check_if_char_already_filled(ch)==false)
            {
                ch_present[c]=ch;
                c++;
            }
        }

    }

    public static int find_how_many_array_cells_filled()
    {
        int c=0;
        for(int i=0; i<94; i++)
        {
            if(ch_present[i] != '\0')
                c++;
        }
        return c;
    }

    public static boolean check_if_char_already_filled(char ch)
    {
        boolean flag=false;
        for(int i=0; i<find_how_many_array_cells_filled(); i++)
        {
            if(ch_present[i]==ch)
            {
                flag=true;
                break;
            }
        }
        return flag;
    }

    public static int find_ch_number(char ch, char[] array)//find the zero-based position of a character in an array. if character not present, return -1 
    {
        int i=0;
        for(i=0; i<array.length; i++)
        {
            if(array[i]==ch)
                break;

            if(i+0==array.length)
                return -1;
        }
        return i;
    }

    public static Character find_ch_at_a_number(int n, char[] array)//inputs the array and the zero-based poisition number and outputs the character in that array at that position
    {
        if(n>array.length+1)
            return null;
        return array[n];
    }

    public static long[][] make_matrix_and_fill_values(char[] array, String text)//make a double dimensional array with the rows and columns as the characters in the ch_present array. Fill in the DD Matrix cells by iterate through all the characters of the text and check if the char of that cell's column comes just next to the character of the row of that cell. return this matrix so formed
    {
        int count=array.length;
        long[][] matrix=new long[count][count];

        for(int i=0; i<text.length()-1; i++)//parse through each position of the text and determine it's pair and 1+ for that pair cell in the matrix
        {
            char first_in_pair=text.charAt(i);
            char second_in_pair=text.charAt(i+1);
            increment_1(first_in_pair, second_in_pair, matrix);
        }
        return matrix;

    }

    public static long[][] increment_1(char row, char column, long[][] matrix)//input the pair(rows and column number) and increment the value in the matrix at that position by one
    {
        matrix[find_ch_number(row, ch_present)][find_ch_number(column, ch_present)]=matrix[find_ch_number(row, ch_present)][find_ch_number(column, ch_present)]+1;
        return matrix;
    }

    public static long pick_random(long n)//random number picker from zero(inclusive) to n(exclusive) 
    {
        return ThreadLocalRandom.current().nextLong(n);
    }

    public static long compute_total(char ch, long[][] matrix)//calculate the sum of all the cells of the row of a character in the matrix 
    {
        long sum=0;
        int j=find_ch_number(ch, ch_present);
        for(int i=0; i<ch_present.length; i++)
            sum=sum+matrix[i][j];
        return sum;
    }

    public static char generate(char starter_char, long[][] matrix)
    {
        int row_of_starter_char=find_ch_number(starter_char, ch_present);
        char generated='\0';
        long total_of_the_row_of_starter_char=compute_total(starter_char, matrix);
        long random=pick_random(total_of_the_row_of_starter_char);
        int i=0;
        long sum=0;
        for(i=0; i<ch_present.length; i++)
        {
            sum=sum+matrix[row_of_starter_char][i];
            if(sum>random)
                break;
        }

        generated=find_ch_at_a_number(i, ch_present);
        return generated;
        
    }
}
