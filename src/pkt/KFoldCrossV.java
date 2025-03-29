package pkt;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

public class KFoldCrossV {

	MomentumBackpropagation mbp;
    int maxEpoch;
    double minHata;
    veriyiBol veri = new veriyiBol();
    private static final File dosya = new File(BP_WithoutMomentum.class.getResource("veriseti.csv").getPath());
    
    public KFoldCrossV(int epoch, double hata, double ogrKatsayisi, double momentum) {
    this.maxEpoch = epoch;
    this.minHata = hata; 
    mbp = new MomentumBackpropagation();
    mbp.setLearningRate(ogrKatsayisi); 
    mbp.setMomentum(momentum);        
    mbp.setMaxIterations(epoch);     
    mbp.setMaxError(hata);           
   
   
    
}  
    public void egitVeTestEt() throws FileNotFoundException {
    	
    	Scanner in = new Scanner(dosya);
    	Scanner scanner = new Scanner(System.in);
    	System.out.print("K değerini girin (kaç fold?): ");
        int k = scanner.nextInt();
        
        
        DataSet dataset = new DataSet(2,1);
        double xMin = 0, xMax = 100; 
        double yMin = 0, yMax = 75; 
        double zMin = 0, zMax = 50; 
        while (in.hasNextLine()) {
            String line = in.nextLine();
            String[] values = line.split(",");
            if (values.length == 3) {
                double x = (Double.parseDouble(values[0]) - xMin) / (xMax - xMin);
                double y = (Double.parseDouble(values[1]) - yMin) / (yMax - yMin);
                double z = (Double.parseDouble(values[2]) - zMin) / (zMax - zMin);
                dataset.add(new DataSetRow(new double[] {x, y}, new double[] {z}));
            }
        }
        dataset.shuffle(); 
        
        
        
        int toplamSatir = dataset.size();
        int foldBoyutu = toplamSatir / k;
        double toplamEgitimHatasi = 0;
        double toplamTestHatasi = 0;
        double ortTestHatasi =0;
        double ortEgitimHatasi=0;
        
        NeuralNetwork<BackPropagation> sinirselAg = 
        		new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 2,15,10, 1);
        sinirselAg.setLearningRule(mbp);
        
        for (int i = 0; i < k; i++) {
        	
            DataSet egitimDs = new DataSet(2,1);
            DataSet testDs = new DataSet(2,1);
            
            
            for (int j = 0; j < toplamSatir; j++) {
                if (j >= i * foldBoyutu && j < (i + 1) * foldBoyutu) {
                    testDs.add(dataset.getRowAt(j));
                } else {
                    egitimDs.add(dataset.getRowAt(j));
                }
            }
        sinirselAg.learn(egitimDs);
        
        for (DataSetRow satir : egitimDs.getRows()) {
            sinirselAg.setInput(satir.getInput());
            sinirselAg.calculate();
            
           
            double tahminNormalized = sinirselAg.getOutput()[0];
            
            
            double tahminOriginal = tahminNormalized * (zMax - zMin) + zMin;
            
            
            double gercekOriginal = satir.getDesiredOutput()[0] * (zMax - zMin) + zMin;

            
            toplamEgitimHatasi += Math.pow(tahminOriginal - gercekOriginal, 2);
        }

       
        ortEgitimHatasi =  toplamEgitimHatasi / egitimDs.size();
        
        
      
        for (DataSetRow satir : testDs.getRows()) {
            sinirselAg.setInput(satir.getInput());
            sinirselAg.calculate();
            
         
            double tahminNormalized = sinirselAg.getOutput()[0];
            
            
            double tahminOriginal = tahminNormalized * (zMax - zMin) + zMin;
            
            
            double gercekOriginal = satir.getDesiredOutput()[0] * (zMax - zMin) + zMin;

            
            toplamTestHatasi += Math.pow(tahminOriginal - gercekOriginal, 2);

           }
        
        ortTestHatasi = toplamTestHatasi / testDs.size();
        }
        sinirselAg.save("ogrenenAg4.nnet");
        System.out.println("\nOrtalama Eğitim Hatası (MSE): " + (ortEgitimHatasi / k));
        System.out.println("Ortalama Test Hatası (MSE): " + (ortTestHatasi / k));
    }
}