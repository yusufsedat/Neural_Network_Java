package pkt;

import java.io.FileNotFoundException;
import java.util.Scanner;

public class program {

	public static void main(String[] args) throws FileNotFoundException {

		Scanner in = new Scanner(System.in);
		
		while(true) {
			System.out.println("  ");
			System.out.println("  ");
			System.out.println("1- Ağı Eğit ve Test Et (Momentumlu) \r\n"
					+ "2- Ağı Eğit ve Test Et (Momentumsuz) \r\n"
					+ "3- Ağı Eğit Epoch Göster \r\n"
					+ "4- Ağı Eğit ve Tekli Test (Momentumlu) \r\n"
					+ "5- K-Fold Test");
			System.out.println("6- Çıkış");
			int secim = in.nextInt();
			
			switch(secim){
			case 1:
				
				BP_WithMomentum ysa = new BP_WithMomentum(1000,0.0001,0.2,0.5);
				ysa.egitVeTestEt();
				break;
				
			case 2: 
				
				BP_WithoutMomentum ysa2 = new BP_WithoutMomentum(1000,0.0001,0.2);
				ysa2.egitVeTestEt();
				break;
			
			case 3:
				BP_WithEpoch ysa3 = new BP_WithEpoch(1000,0.0001,0.2,0.5);
				ysa3.egitVeTestEtEpochBazli();
				break;
				
			case 4:
				BP_Single ysa4 = new BP_Single(1000,0.0001,0.2,0.5);
				ysa4.egitVeTestEt();
				break;
				
			case 5:
				KFoldCrossV ysa5 = new KFoldCrossV(1000,0.0001,0.2,0.5);
				ysa5.egitVeTestEt();
				break;
				
			case 6:				
				in.close(); 
                System.exit(0); 
				break;
				
			default:
				System.out.println("Geçersiz bir seçenek girdiniz. Lütfen tekrar deneyin.");
			}
			
		}
		
		

}
}