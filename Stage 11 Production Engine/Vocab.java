import java.io.*;
import java.util.*;

public class Vocab implements Serializable 
{
    private static final long serialVersionUID = 1L;
    public static final char UNK_CHAR = '\u25A1'; // White square symbol for unknown

    private char[] itos;
    private Map<Character, Integer> stoi;

    public Vocab(char[] characters) 
    {
        this.itos = characters.clone();
        this.stoi = new HashMap<>();
        for (int i = 0; i < itos.length; i++) 
        {
            stoi.put(itos[i], i);
        }
    }

    public static Vocab buildFromText(String text, int minFreq) 
    {
        Map<Character, Integer> counts = new HashMap<>();
        for (int i = 0; i < text.length(); i++) 
        {
            char c = text.charAt(i);
            counts.put(c, counts.getOrDefault(c, 0) + 1);
        }

        // Build list: index 0 is always UNK_CHAR
        List<Character> charList = new ArrayList<>();
        charList.add(UNK_CHAR);

        // Sort characters: most frequent first
        List<Map.Entry<Character, Integer>> sortedEntries = new ArrayList<>(counts.entrySet());
        sortedEntries.sort((a, b) -> b.getValue().compareTo(a.getValue()));

        for (Map.Entry<Character, Integer> entry : sortedEntries) 
        {
            if (entry.getValue() >= minFreq && entry.getKey() != UNK_CHAR) 
            {
                charList.add(entry.getKey());
            }
        }

        char[] chars = new char[charList.size()];
        for (int i = 0; i < chars.length; i++) 
        {
            chars[i] = charList.get(i);
        }
        return new Vocab(chars);
    }

    public int encode(char c) 
    {
        Integer idx = stoi.get(c);
        if (idx == null) 
        {
            return 0; // Return index of UNK_CHAR
        }
        return idx;
    }

    public char decode(int idx) 
    {
        if (idx < 0 || idx >= itos.length) 
        {
            return UNK_CHAR;
        }
        return itos[idx];
    }

    public int[] encodeString(String s) 
    {
        int[] tokens = new int[s.length()];
        for (int i = 0; i < s.length(); i++) 
        {
            tokens[i] = encode(s.charAt(i));
        }
        return tokens;
    }

    public String decodeTokens(int[] tokens) 
    {
        StringBuilder sb = new StringBuilder();
        for (int t : tokens) 
        {
            sb.append(decode(t));
        }
        return sb.toString();
    }

    public int size() 
    {
        return itos.length;
    }

    public char[] getCharacters() 
    {
        return itos.clone();
    }
}
