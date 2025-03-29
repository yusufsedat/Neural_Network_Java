package pkt;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

public class veriyiBol {
	
	private static final File dosya = new File(BP_WithoutMomentum.class.getResource("veriseti.csv").getPath());
	
    public DataSet[] veriyiBol() throws FileNotFoundException {
        Scanner in = new Scanner(dosya);
        DataSet tumVeriSeti = new DataSet(2, 1);

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
                tumVeriSeti.add(new DataSetRow(new double[] {x, y}, new double[] {z}));
            }
        }

        tumVeriSeti.shuffle();
        int toplamSatir = tumVeriSeti.size();
        int egitimSayisi = (int) (toplamSatir * 0.75);

        DataSet egitimDs = new DataSet(2, 1);
        DataSet testDs = new DataSet(2, 1);

        for (int i = 0; i < toplamSatir; i++) {
            if (i < egitimSayisi) {
                egitimDs.add(tumVeriSeti.getRowAt(i));
            } else {
                testDs.add(tumVeriSeti.getRowAt(i));
            }
        }

        return new DataSet[] {egitimDs, testDs};
    }
}
