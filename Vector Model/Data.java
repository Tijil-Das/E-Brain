import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Data 
{    
    public static String chunk1 = load_text();

    public static String load_text()
    {
        //Try possible locations for Text.txt so it works in terminal, IDE, or subfolder
        String[] possible_paths = {
            "Text.txt",
            "../Assets/Text.txt",
            "Assets/Text.txt",
            "../../Assets/Text.txt"
        };

        for(int i = 0; i < possible_paths.length; i++)
        {
            try
            {
                Path p = Paths.get(possible_paths[i]);
                if(Files.exists(p))
                {
                    return Files.readString(p);
                }
            }
            catch(Exception e)
            {
                //Try next path
            }
        }
        return "";
    }
}
