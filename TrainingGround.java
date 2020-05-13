package programmingProject;

public class TrainingGround {
	public static void main(String[] args) {
		
		int[] layers = new int[] {5, 4, 3};
		NeuralNetwork n = new NeuralNetwork(layers);
		float[] inputs = new float[] {0.8f, 0.3f, 0.6f, 0.0f, 0.5f};
		float[] targets = new float[] {0.2f, 0.7f, 0.0f};
		
		for(int i = 0; i < 500; i++) {
			
			n.train(inputs, targets);
			float[] output = n.getOutputs();
			for(int j = 0; j < output.length; j++) {
				
				System.out.println(output[j]);
				
			}
			System.out.println("");
		}
		

	}
}
