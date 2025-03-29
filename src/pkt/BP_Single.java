package pkt;

import java.io.FileNotFoundException;
import java.util.Locale;
import java.util.Scanner;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.nnet.learning.MomentumBackpropagation;

public class BP_Single {
	MomentumBackpropagation mbp;
    int maxEpoch;
    double minHata;
    veriyiBol veri = new veriyiBol();
    
    public BP_Single(int epoch, double hata, double ogrKatsayisi, double momentum) {
        this.maxEpoch = epoch;
        this.minHata = hata; 
        mbp = new MomentumBackpropagation();
        mbp.setLearningRate(ogrKatsayisi); 
        mbp.setMomentum(momentum);        
        mbp.setMaxIterations(epoch);     
        mbp.setMaxError(hata);          
        
        
    }
    public void egitVeTestEt() throws FileNotFoundException {
    	
       
        DataSet[] veriSetleri = veri.veriyiBol();
        DataSet egitimDs = veriSetleri[0];
        

        
        NeuralNetwork<BackPropagation> sinirselAg = 
                new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 2,15,10, 1);
        sinirselAg.setLearningRule(mbp);

        
        sinirselAg.learn(egitimDs);
        sinirselAg.save("ogrenenAg3.nnet");
        System.out.println("Eğitim tamamlandı.");
        
        
        double zMin = 0; 
        double zMax = 50; 
        
        int secim=0;
        do {
        	Scanner scanner = new Scanner(System.in).useLocale(Locale.US);
            System.out.print("Oksijen seviyesi girin (0-100): ");
            double oksijen = scanner.nextDouble();
            System.out.print("Yağış durumu girin (0-75): ");
            double yagis = scanner.nextDouble();
            
            double xMin = 0, xMax = 100;
            double yMin = 0, yMax = 75;
            double oksijenNormalized = (oksijen - xMin) / (xMax - xMin);
            double yagisNormalized = (yagis - yMin) / (yMax - yMin);
            
            sinirselAg.setInput(oksijenNormalized, yagisNormalized);
            sinirselAg.calculate();
            double tahminNormalized = sinirselAg.getOutput()[0];            
           
            double tahmin = tahminNormalized * (zMax - zMin) + zMin;
            System.out.println("Tahmin edilen değer: " + tahmin);
        	
            System.out.println("0 ile programdan çıkabilirsiniz."
            		+ "Başka değerler girmek için herhangi sayı girin.");
            secim = scanner.nextInt();
            
        }while(secim != 0);
        
       }
}

