import com.aliasi.classify.LMClassifier;
import com.aliasi.util.Files;

import com.aliasi.classify.Classification;
import com.aliasi.classify.Classified;
import com.aliasi.classify.DynamicLMClassifier;

import com.aliasi.lm.NGramProcessLM;

import java.io.*;

public class PolarityBasic {

    File mPolarityDir; // directory che contiene il dataset
    String[] mCategories; // categorie prese dalle sottocartelle (pos e neg)
    DynamicLMClassifier<NGramProcessLM> mClassifier; // the dynamic LM classifiers produced by the factory method createNGramProcess() deserialize with language models that are instances of LanguageModel.Process
                                                     // la classe NGramProcessLM è una classe che implementa l'interfaccia LanguageModel.Process

    PolarityBasic(String[] args) {
        System.out.println("\nBASIC POLARITY DEMO");
        mPolarityDir = new File(args[0],"txt_sentoken");
        System.out.println("\nData Directory=" + mPolarityDir);
        mCategories = mPolarityDir.list();
        for(String tmp : mCategories){
            System.out.println(tmp);
        }
        int nGram = 8;
        mClassifier 
            = DynamicLMClassifier
            .createNGramProcess(mCategories,nGram);
    }

    void run() throws ClassNotFoundException, IOException {
        train();
        evaluate();
    }

    boolean isTrainingFile(File file) {
        return file.getName().contains("9");  // test on fold 9
    }

    void train() throws IOException {
        int numTrainingCases = 0;
        int numTrainingChars = 0;
        System.out.println("\nTraining.");
        for (int i = 0; i < mCategories.length; ++i) {
            String category = mCategories[i];
            Classification classification
                = new Classification(category);
            File file = new File(mPolarityDir,mCategories[i]);
            File[] trainFiles = file.listFiles();
            for (int j = 0; j < trainFiles.length; ++j) {
                File trainFile = trainFiles[j];
                if (isTrainingFile(trainFile)) {
                    ++numTrainingCases;
                    String review = Files.readFromFile(trainFile,"ISO-8859-1");
                    numTrainingChars += review.length();
                    Classified<CharSequence> classified
                        = new Classified<CharSequence>(review,classification);
                    mClassifier.handle(classified);
                }
            }
        }

        // compile the model into a file
        compile();
        System.out.println("  # Training Cases=" + numTrainingCases);
        System.out.println("  # Training Chars=" + numTrainingChars);
    }

    void compile() throws FileNotFoundException, IOException {
            System.out.println("\nCompiling.\n  Model file=subjectivity.model");
            FileOutputStream fileOut = new FileOutputStream("subjectivity.model");
            ObjectOutputStream objOut = new ObjectOutputStream(fileOut);
            mClassifier.compileTo(objOut);
            objOut.close();
    }

    void evaluate() throws IOException {
        System.out.println("\nEvaluating.");
        int numTests = 0;
        int numCorrect = 0;
        for (int i = 0; i < mCategories.length; ++i) {
            String category = mCategories[i];
            File file = new File(mPolarityDir,mCategories[i]);
            File[] trainFiles = file.listFiles();
            for (int j = 0; j < trainFiles.length; ++j) {
                File trainFile = trainFiles[j];
                if (!isTrainingFile(trainFile)) {
                    String review = Files.readFromFile(trainFile,"ISO-8859-1");
                    ++numTests;
                    Classification classification
                        = mClassifier.classify(review);
                    if (classification.bestCategory().equals(category))
                        ++numCorrect;
                }
            }
        }
        System.out.println("  # Test Cases=" + numTests);
        System.out.println("  # Correct=" + numCorrect);
        System.out.println("  % Correct=" 
                           + ((double)numCorrect)/(double)numTests);
    }

    static void test() throws Exception{
        FileInputStream fis = new FileInputStream("subjectivity.model");
        ObjectInputStream in = new ObjectInputStream(fis);
        String test = "I missed the New Moon trailer..."; // negative tweet
        LMClassifier classifier = (LMClassifier) in.readObject();
        System.out.println(classifier.classify(test).bestCategory());

    }

    public static void main(String[] args) {
        try {
            new PolarityBasic(args).run();
            // test();
        } catch (Throwable t) {
            System.out.println("Thrown: " + t);
            t.printStackTrace(System.out);
        }
    }

}

