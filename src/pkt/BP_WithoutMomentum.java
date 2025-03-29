package pkt;


import java.io.FileNotFoundException;


import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

public class BP_WithoutMomentum {   
    
    BackPropagation bp;
    int maxEpoch;
    double minHata;
    veriyiBol veri = new veriyiBol();
    
    public BP_WithoutMomentum(int epoch, double hata, double ogrKatsayisi) {
        this.maxEpoch = epoch;
        this.minHata = hata; 
        bp = new BackPropagation();
        bp.setLearningRate(ogrKatsayisi);
        bp.setMaxIterations(epoch);
        bp.setMaxError(hata);
        
        
    }

    public void egitVeTestEt() throws FileNotFoundException {
       
        DataSet[] veriSetleri = veri.veriyiBol();
        DataSet egitimDs = veriSetleri[0];
        DataSet testDs = veriSetleri[1];

        
        NeuralNetwork<BackPropagation> sinirselAg = 
                new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 2,15,10, 1);
        sinirselAg.setLearningRule(bp);

        
        sinirselAg.learn(egitimDs);
        sinirselAg.save("ogrenenAg.nnet");
        System.out.println("Eğitim tamamlandı.");

        
        double zMin = 0; 
        double zMax = 50; 
        double toplamEgitimHatasi = 0;
        for (DataSetRow satir : egitimDs.getRows()) {
            sinirselAg.setInput(satir.getInput());
            sinirselAg.calculate();
            
            
            double tahminNormalized = sinirselAg.getOutput()[0];
            
            
            double tahminOriginal = tahminNormalized * (zMax - zMin) + zMin;
            
            
            double gercekOriginal = satir.getDesiredOutput()[0] * (zMax - zMin) + zMin;

            
            toplamEgitimHatasi += Math.pow(tahminOriginal - gercekOriginal, 2);
        }

        
        System.out.println("Eğitim Hatası (MSE): " + toplamEgitimHatasi / egitimDs.size());
        
        
        double toplamHata = 0;
        for (DataSetRow satir : testDs.getRows()) {
            sinirselAg.setInput(satir.getInput());
            sinirselAg.calculate();
            
            
            double tahminNormalized = sinirselAg.getOutput()[0];
            
            
            double tahminOriginal = tahminNormalized * (zMax - zMin) + zMin;
            
            
            double gercekOriginal = satir.getDesiredOutput()[0] * (zMax - zMin) + zMin;

            
            toplamHata += Math.pow(tahminOriginal - gercekOriginal, 2);

          
        }
               
        System.out.println("Test Hatası (MSE): " + toplamHata / testDs.size());
    }

   


}

