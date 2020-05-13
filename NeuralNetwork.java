package programmingProject;

import java.util.Random;

/**
 * This class contains the basic structure of a feedForward
 * neural network that learns using backpropagation.
 * @author 02asswid
 *
 */

public class NeuralNetwork {
	
	//	Variables
	private int layerCount;	//	Number of layers.
	private int[] layers;	//	Number of neurons in each layer.
	private float[][] neurons;	// The values held by every neuron
	private float[][] sums;	//	The sums of all weighted connections and biases of every neuron
	private float[][] biases;	//	The biases of every neuron
	private float[][][] weights;	// The weighted connections between neurons of two adjacent layers
	private float[][] neuronDerivatives;	//	The derivatives of the error with respect to each neuron
	private float[][] biasDerivatives;
	private float[][][] weightDerivatives;
	public float[] targetOut;
	
	private float learningRate;	//	How big 'steps' the network takes during training
	
	//	Methods
	
	/**
	 * Defines various structures of the network.
	 * @param layers An array containing the length of each layer in the network.
	 */
	NeuralNetwork(int[] layers) {	//	Constructor

		//	Assigns values to variables
		layerCount = layers.length;
		this.layers = layers;
		neurons = new float[layerCount][];	// The size of the neurons array is one step bigger in size
		sums = new float[layerCount - 1][];	//	As the other arrays do not include the input layer
		biases = new float[layerCount - 1][];	// Leading to diffrent indicies for tracking 'the same' thing
		weights = new float[layerCount - 1][][];	//	E.g. the bias of the neuron at neurons[i][j] is biases[i - 1][j]
		neuronDerivatives = new float[layerCount - 1][];
		biasDerivatives = new float[layerCount - 1][];
		weightDerivatives = new float[layerCount - 1][][];
		targetOut = new float[layers[layerCount - 1]];
		
		initNeurons();
		initSums();
		initBiases();
		initWeights();
		initNeuronDerivatives();
		initBiasDerivatives();
		initWeightDerivatives();
		
		learningRate = 2f;	//	Used to make the network take bigger steps in learning (may lead to the network overshooting its goal if set to be too high)
		
	}
	
	private void initNeurons() { // Generates neurons
		
		for(int i = 0; i < layerCount; i++) {	// Iterates through every layer
			
			neurons[i] = new float[layers[i]];
			
		}
		
	}
	
	private void initSums() {
		
		for(int i = 0; i < layerCount - 1; i++) {
			
			sums[i] = new float[layers[i]];
			
		}
		
	}
	
	private void initBiases() {	//	Generates biases randomly
		
		Random rnd = new Random();
		
		for(int i = 0; i < layerCount - 1; i++) {	//	iterates through every layer except for the input layer
			
			int layerSize = layers[i + 1];	//	Number of neurons in the current layer
			biases[i] = new float[layerSize];
			for(int j = 0; j < layerSize; j++) {	//	Iterates through every neuron in the current layer
				
				biases[i][j] = 16 * rnd.nextFloat() - 8; // Assigns the given bias a value between -8 and 8
				
			}
			
		}
		
	}
	
	private void initWeights() { //	Generates weights randomly
		
		Random rnd = new Random();
		
		for(int i = 0; i < layerCount - 1; i++) {
			
			int layerSize = layers[i + 1];
			int layerSizePrev = layers[i];	//	Number of neurons in the previous layer
			weights[i] = new float[layerSize][];
			for(int j = 0; j < layerSize; j++) {
				
				weights[i][j] = new float[layerSizePrev];
				for(int k = 0; k < layerSizePrev; k++) {	// Iterates through every neuron in the previous layer
					
					weights[i][j][k] = 4 * rnd.nextFloat() - 2;	//	Assigns the given weight a value between -2 and 2
					
				}
				
			}
			
		}
		
	}
	
	private void initNeuronDerivatives(){	//	Generates neuron derivaitves
		
		for(int i = 0; i < layerCount - 1; i++) {
			
			neuronDerivatives[i] = new float[layers[i]];
			
		}
		
	}
	
	private void initBiasDerivatives() {	//	Generates bias derivaitves
		
		for(int i = 0; i < layerCount - 1; i++) {
			
			biasDerivatives[i] = new float[layers[i + 1]];
			
		}
		
	}
	
	private void initWeightDerivatives() {	//	Generates weight derivaitves
		
		for(int i = 0; i < layerCount - 1; i++) {
			
			weightDerivatives[i] = new float[layers[i + 1]][];
			for(int j = 0; j < layers[i + 1]; j++) {
				
				weightDerivatives[i][j] = new float[layers[i]];
				
			}
			
		}
		
	}
	/**
	 * Accesses the current output of the network.
	 * @return Returns the values held by the neurons in the output layer.
	 */
	public float[] getOutputs() {
		
		return neurons[layerCount - 1];
		
	}
	/**
	 * Takes a set of inputs and feeds them into the network, producing a set of outputs.
	 * @param inputs A float array with the same length as the input-layer.
	 */
	public void feedForward(float[] inputs) {
		
		for(int i = 0; i < layers[0]; i++) {
			
			neurons[0][i] = inputs[i];	//	Sets every neuron in the input layer to the corresponding input 
			
		}
		for(int i = 1; i < layerCount; i++) {
			
			for(int j = 0; j < layers[i]; j++) {
				
				float inputSum = 0;	//	The sum of all connections to the given neuron from the previous layer
				
				int layerSizePrev = layers[i - 1];
				
				for(int k = 0; k < layerSizePrev; k++) {
					
					inputSum += weights[i - 1][j][k] * neurons[i - 1][k];	//	Adds the value of a weighted connection
					
				}
				
				inputSum += biases[i - 1][j];	//	Adds the bias of the neuron
				sums[i - 1][j] = inputSum;	// Sets the corresponding sum to the input sum
				neurons[i][j] = sigmoid(inputSum);	//	Takes the sum through a sigmoid function and sets the value of the neuron to the given result.
				
			}
			
		}						
	
	}
	/**
	 * Trains the network on one set of inputs and desired outputs.
	 * @param inputs The inputs that get fed into the network.
	 * @param targetOut The desired output of the network.
	 */
	public void train(float[] inputs, float[] targetOut) {
		
		feedForward(inputs);
		for(int i = 0; i < layers[layerCount - 1]; i++) {
			
			this.targetOut[i] = targetOut[i];
			
		}
		findDerivatives();
		tweakWeights();
		tweakBiases();
		
	}

	private void findNeuronDerivatives() {	//	Finds the derivative of the error with respect to each neuron individually (excluding the input neurons)
		
		findOutputNeuronDerivative();
		findHiddenNeuronDerivative();
		
	}
	private void findOutputNeuronDerivative() {	//	Finds all derivatives with respect to output layer neurons
		
		for(int i = 0; i < layers[layerCount - 1]; i++) {	//	Iterates through every neuron in the output layer 
			
			neuronDerivatives[layerCount - 2][i] = 2 * (neurons[layerCount - 1][i] - targetOut[i]);
			
		}
		
	}
	private void findHiddenNeuronDerivative() {	//	Finds all derivatives with respect to hidden layer neurons
		
		for(int i = layerCount - 3; i >= 0; i--) {	//	Iterates through every hidden layer
			
			for(int j = 0; j < layers[i]; j++) {	//	Iterates through every neuron in the given hidden layer
				
				float neuronDerivative = 0;
				for(int k = 0; k < layers[i + 1]; k++) {	//	Iterates through every neuron in the next layer
					
					neuronDerivative += neuronDerivatives[i + 1][k] * sigmoid(sums[i + 1][k]) * (1 - sigmoid(sums[i + 1][k])) * weights[i][k][j];
					
				}
				neuronDerivatives[i][j] = neuronDerivative;
				
			}
			
		}
		
	}
	
	private void findWeightDerivatives() {	// Finds all derivatives with respect to weights
		
		for(int i = layerCount - 2; i >= 0; i--) {
			
			for(int j = 0; j < layers[i + 1]; j++) {
				
				for(int k = 0; k < layers[i]; k++) {
					
					weightDerivatives[i][j][k] = neuronDerivatives[i][j] * sigmoid(sums[i][j]) * (1 - sigmoid(sums[i][j])) * neurons[i][k];
					
				}
				
			}
			
		} 
		
	}
	
	private void findBiasDerivatives() {	//	Finds all derivatives with respect to biases
		
		for(int i = layerCount - 2; i >= 0; i--) {
			
			for(int j = 0; j < layers[i + 1]; j++) {
				
				biasDerivatives[i][j] = neuronDerivatives[i][j] * sigmoid(sums[i][j]) * (1 - sigmoid(sums[i][j]));
				
			}
			
		} 
		
	}
	
	private void findDerivatives() {
		
		findNeuronDerivatives();
		findWeightDerivatives();
		findBiasDerivatives();
		
	}
	
	private void tweakWeights() {
		
		for(int i = 0; i < layerCount - 1; i++) {
			
			for(int j = 0; j < layers[i + 1]; j++) {
				
				for(int k = 0; k < layers[i]; k++) {			
						
						weights[i][j][k] -= weightDerivatives[i][j][k] * learningRate;						
				
				}
				
			}
			
		}
		
	}
	
	private void tweakBiases() {
		
		for(int i = 0; i < layerCount - 1; i++) {
			
			for(int j = 0; j < layers[i + 1]; j++) {
					
					biases[i][j] -= biasDerivatives[i][j] * learningRate;
				
			}
			
		}
		
	}
	/**
	 * 
	 * @param input The number to be taken taken through the function.
	 * @return Returns the value of the input taken through the sigmoid-funtion (a number between 0 and 1).
	 */
	private float sigmoid(float input) {
		
		return 1 / ((float)Math.exp(-input) + 1);
		
	}
}