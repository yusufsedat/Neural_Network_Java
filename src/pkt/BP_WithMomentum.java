package pkt;


import java.io.FileNotFoundException;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.nnet.learning.MomentumBackpropagation;

public class BP_WithMomentum {

	MomentumBackpropagation mbp;
    int maxEpoch;
    double minHata;
    veriyiBol veri = new veriyiBol();
    
    public BP_WithMomentum(int epoch, double hata, double ogrKatsayisi, double momentum) {
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
        DataSet testDs = veriSetleri[1];

        
        NeuralNetwork<BackPropagation> sinirselAg = 
                new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 2,15,10,1);
        sinirselAg.setLearningRule(mbp);

        
        sinirselAg.learn(egitimDs);
        sinirselAg.save("ogrenenAg2.nnet");
        System.out.println("Eğitim tamamlandı.");

        
        double zMin = 0; 
        double zMax = 50;
        double toplamEgitimHatasi = 0;
        double toplamTestHatasi = 0;
        
        for (DataSetRow satir : egitimDs.getRows()) {
            sinirselAg.setInput(satir.getInput());
            sinirselAg.calculate();
            
            
            double tahminNormalized = sinirselAg.getOutput()[0];
                        
            double tahminOriginal = tahminNormalized * (zMax - zMin) + zMin;
            
            double gercekOriginal = satir.getDesiredOutput()[0] * (zMax - zMin) + zMin;

            
            toplamEgitimHatasi += Math.pow(tahminOriginal - gercekOriginal, 2);
        }

        
        System.out.println("Eğitim Hatası (MSE): " + toplamEgitimHatasi / egitimDs.size());
        
               
        for (DataSetRow satir : testDs.getRows()) {
            sinirselAg.setInput(satir.getInput());
            sinirselAg.calculate();
            
            
            double tahminNormalized = sinirselAg.getOutput()[0];
            
            
            double tahminOriginal = tahminNormalized * (zMax - zMin) + zMin;
            
            
            double gercekOriginal = satir.getDesiredOutput()[0] * (zMax - zMin) + zMin;

            
            toplamTestHatasi += Math.pow(tahminOriginal - gercekOriginal, 2);
            
        }
        
        System.out.println("Test Hatası (MSE): " + toplamTestHatasi / testDs.size());
    }
}
