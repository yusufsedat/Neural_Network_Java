package pkt;

import java.io.FileNotFoundException;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.nnet.learning.MomentumBackpropagation;

public class BP_WithEpoch {
	
	MomentumBackpropagation mbp;
    int maxEpoch;
    double minHata;
    veriyiBol veri = new veriyiBol();
    
    public BP_WithEpoch(int epoch, double hata, double ogrKatsayisi, double momentum) {
        this.maxEpoch = epoch;
        this.minHata = hata; 
        mbp = new MomentumBackpropagation();
        mbp.setLearningRate(ogrKatsayisi); 
        mbp.setMomentum(momentum);       
        mbp.setMaxIterations(epoch);     
        mbp.setMaxError(hata);          
        
        
    }
    public void egitVeTestEtEpochBazli() throws FileNotFoundException {
    	
       
        DataSet[] veriSetleri = veri.veriyiBol();
        DataSet egitimDs = veriSetleri[0];
        DataSet testDs = veriSetleri[1];

        
        
        NeuralNetwork<BackPropagation> sinirselAg = 
                new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 2,15,10, 1);
        sinirselAg.setLearningRule(mbp);
        
        
        double zMin = 0; 
        double zMax = 50; 
        
        
        for (int epoch = 1; epoch <= maxEpoch; epoch++) {
           
            sinirselAg.learn(egitimDs);

            
            double toplamEgitimHatasi = 0;
            for (DataSetRow satir : egitimDs.getRows()) {
                sinirselAg.setInput(satir.getInput());
                sinirselAg.calculate();
             
                double tahminNormalized = sinirselAg.getOutput()[0];
                
               
                double tahminOriginal = tahminNormalized * (zMax - zMin) + zMin;
                
               
                double gercekOriginal = satir.getDesiredOutput()[0] * (zMax - zMin) + zMin;
                toplamEgitimHatasi += Math.pow(tahminOriginal - gercekOriginal, 2);
            }
            double ortalamaEgitimHatasi = toplamEgitimHatasi / egitimDs.size();

           
            double toplamTestHatasi = 0;
            for (DataSetRow satir : testDs.getRows()) {
                sinirselAg.setInput(satir.getInput());
                sinirselAg.calculate();
                double tahminNormalized = sinirselAg.getOutput()[0];
                
                
                double tahminOriginal = tahminNormalized * (zMax - zMin) + zMin;
                
                
                double gercekOriginal = satir.getDesiredOutput()[0] * (zMax - zMin) + zMin;

                
                toplamTestHatasi += Math.pow(tahminOriginal - gercekOriginal, 2);
            }
            double ortalamaTestHatasi = toplamTestHatasi / testDs.size();

            
            System.out.println("Epoch: " + epoch);
            System.out.println("Eğitim Hatası (MSE): " + ortalamaEgitimHatasi);
            System.out.println("Test Hatası (MSE): " + ortalamaTestHatasi);
        }
    }
}
