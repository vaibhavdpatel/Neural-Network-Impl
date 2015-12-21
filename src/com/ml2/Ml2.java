/**
 * 
 */
package com.ml2;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;


/**
 * @author vaibhav
 *
 */
public class Ml2 {


	private static String posClass, negClass;

	private static HashMap<Instance, InstanceInfo> hash = 
			new HashMap<Instance, InstanceInfo> ();

	private static HashMap<Instance, Instance> hashOrigToRandom = 
			new HashMap<Instance, Instance> ();

	private static HashMap<Instance, Instance> hashNeg = 
			new HashMap<Instance, Instance> ();

	private static HashMap<Instance, Instance> hashPos = 
			new HashMap<Instance, Instance> ();

	/*private static final String ARFF_FILE_PATH = 
			"/afs/cs.wisc.edu/u/v/a/vaibhav/private/"
					+ "Eclipse/ML2/src/inputfiles/sonar.arff";
	 */
	double predict(Instance instance, double[] weights) {

		int size = instance.numAttributes()-1;
		double net = weights[size] * 1; // for bias = last node
		for(int i = 0; i < size; i++) {
			net += instance.value(i) * weights[i];
		}
		//System.out.print("\nnet: " + net);
		net = 1/(1 + Math.exp(-net));
		//System.out.print(" " + net + "\n");
		return net;
	}

	void setClassLabels (Instances instances) {
		String s = instances.attribute(instances.numAttributes()-1).toString();
		s = s.substring(s.indexOf("{"));

		negClass = s.substring(1, s.indexOf(","));
		posClass = s.substring(s.indexOf(",") + 1, s.length()-1);
	}

	void initializeWeights (double[] weights, boolean random) {

		int size = weights.length;

		if(random == true) {
			Random r = new Random();
			int sign = 0;
			for(int i=0; i < size; i++) {

				sign = r.nextInt()%2;
				weights[i] = r.nextDouble();

				weights[i] = (sign == 1) ? weights[i]*1 : weights[i]*-1;
				//System.out.print(weights[i] + " ");
			} 
		} else {
			for(int i=0; i < size; i++) {
				weights[i] = 0.1;
			}
		}
	}

	void backpropagate(Instance instance, double[] weights, 
			double out, double y, double learningRate) {

		//double error = 0.5 * Math.pow((y - out), 2);
		double doeE_by_doeOut = (out - y);
		double doeOut_by_doeNet = out * (1 - out);
		int size = instance.numAttributes() - 1;
		double doeE_by_doeWi = 1;
		double temp = doeE_by_doeOut * doeOut_by_doeNet;

		for(int i = 0; i < size; i++) {
			doeE_by_doeWi = temp * (-1) * instance.value(i);
			weights[i] += doeE_by_doeWi * learningRate;
		}

		weights[size] += temp * - 1 * learningRate; 
	}

	/**
	 * Stratifies the Instances Set.
	 * 
	 * @param dataInstances
	 * @param folds
	 * @return
	 */
	Instances[] stratifySet (Instances dataInstances, int folds) {

		int t = dataInstances.numAttributes()-1;

		dataInstances.setClassIndex(t);

		int num = dataInstances.numInstances();

		for(int i = 0; i < num; i++) {

			InstanceInfo instanceInfo = new InstanceInfo();
			instanceInfo.setActualIndex(i);

			hash.put(dataInstances.instance(i), instanceInfo);
		}

		Random r = new Random();
		dataInstances.randomize(r);

		Instances posInstances = new Instances(dataInstances, 0);
		Instances negInstances = new Instances(dataInstances, 0);

		for(int i = 0; i < num; i++) {

			if(dataInstances.instance(i).stringValue(t).equals(posClass)) {
				posInstances.add(dataInstances.instance(i));
				hashPos.put(posInstances.instance(posInstances.numInstances()-1), dataInstances.instance(i));
				//System.out.println(dataInstances.instance(i) + "\n" + posInstances.instance(posInstances.numInstances()-1) + "\n");
			} else {
				negInstances.add(dataInstances.instance(i));
				hashNeg.put(negInstances.instance(negInstances.numInstances()-1), dataInstances.instance(i));
			}

		}

		int numOfNegs = negInstances.numInstances() / folds;
		int numOfPos = posInstances.numInstances() / folds;

		/*System.out.println(posInstances.numInstances() + " " + 
				negInstances.numInstances());

		System.out.println(numOfNegs + " " + 
				numOfPos);
		 */
		Instances[] stratifiedInstances = new Instances [folds];
		for(int l = 0; l < folds; l++) {
			stratifiedInstances[l] = new Instances(dataInstances,0);
		}

		int n = 0, p = 0;

		for(int i = 0; i < folds; i++) {
			for(int j = 0; j < numOfNegs; j++) {
				stratifiedInstances[i].add(negInstances.instance(n));

				InstanceInfo instanceInfo = hash.get(hashNeg.get(negInstances.instance(n)));
				instanceInfo.setStratifiedIndex(i);
				hash.remove(hashNeg.get(negInstances.instance(n)));
				hash.put(hashNeg.get(negInstances.instance(n)), instanceInfo);
				n++;
			}


			for(int j = 0; j < numOfPos; j++) {
				stratifiedInstances[i].add(posInstances.instance(p));

				InstanceInfo instanceInfo = hash.get(hashPos.get(posInstances.instance(p)));
				//System.out.print (instanceInfo.getActualIndex() + " ");
				instanceInfo.setStratifiedIndex(i);
				hash.remove(hashPos.get(posInstances.instance(p)));
				hash.put(hashPos.get(posInstances.instance(p)), instanceInfo);
				p++;
			}

		}

		int leftNegs = negInstances.numInstances() - numOfNegs * folds;
		int leftPos = posInstances.numInstances() - numOfPos * folds;

		for(int j = 0; j < leftNegs; j++) {
			int temp = numOfNegs * folds + j;
			stratifiedInstances[j].add(negInstances.instance(temp));

			InstanceInfo instanceInfo = hash.get(hashNeg.get(negInstances.instance(temp)));
			instanceInfo.setStratifiedIndex(j);
			hash.remove(hashNeg.get(negInstances.instance(temp)));
			hash.put(hashNeg.get(negInstances.instance(temp)), instanceInfo);
		}

		for(int j = leftNegs; j < leftNegs + leftPos; j++) {
			int temp = numOfPos * folds + j -leftNegs ;

			InstanceInfo instanceInfo = hash.get(hashPos.get(posInstances.instance(temp)));
			instanceInfo.setStratifiedIndex(j);
			hash.remove(hashPos.get(posInstances.instance(temp)));
			hash.put(hashPos.get(posInstances.instance(temp)), instanceInfo);

			stratifiedInstances[j].add(posInstances.instance(temp));
		}

		for(int i = 0; i < folds; i++) {
			stratifiedInstances[i].randomize(r);
		}

		for(int i = 0; i < dataInstances.numInstances(); i++) {
			//System.out.println( hash.get(dataInstances.instance(i)).getStratifiedIndex() );
		}

		return stratifiedInstances;
	}

	public static void main(String[] args) throws IOException {

		if(args.length != 4) {
			System.out.println("very few args");
			return;
		}

		File datasetFile = new File(args[0]);
		int folds = Integer.parseInt(args[1]);
		double learningRate = Double.parseDouble(args[2]);
		int times = Integer.parseInt(args[3]);

		Ml2 ml2 = new Ml2();
		ArffLoader arffLoader = new ArffLoader();

		arffLoader.setFile(datasetFile);

		Instances dataInstances = arffLoader.getDataSet();
		Instances origInstances = new Instances(dataInstances);

		for(int i=0; i < dataInstances.numInstances(); i++) {
			hashOrigToRandom.put(origInstances.instance(i), dataInstances.instance(i));
		}

		double[] weights = new double[dataInstances.numAttributes()];

		int classIndex = dataInstances.numAttributes()-1;
		double y ;

		Instances[] stratifiedInstances = new Instances [folds];
		for(int l = 0; l < folds; l++) {
			stratifiedInstances[l] = new Instances(dataInstances,0);
		}

		ml2.initializeWeights(weights, false);
		ml2.setClassLabels(dataInstances);
		stratifiedInstances = ml2.stratifySet(dataInstances, folds);

		for(int f = 0; f < folds; f ++) {

			// Training part
			for(int t = 0; t < times; t++) {
				for(int curr = 0; curr < folds; curr++) {
					if(f != curr) {
						for(int i = 0; i < stratifiedInstances[curr].numInstances(); i++) {
							Instance instance = stratifiedInstances[curr].instance(i);
							double out = ml2.predict(instance, weights);

							if(instance.stringValue(classIndex).equals(posClass)) {
								y = 0.9999;
							} else {
								y = 0.0001;
							}

							ml2.backpropagate(instance, weights, out, y, learningRate);
						}
					}
				}
			}
		}
		
		//int trueNeg = 0, truePos = 0, falseNeg = 0, falsePos = 0;

		// prediction + accuracy calculation part
		for(int i = 0; i < origInstances.numInstances(); i++) {
			Instance instance = origInstances.instance(i);
			double out = ml2.predict(instance, weights);

			Instance instance2 = hashOrigToRandom.get(origInstances.instance(i));
			InstanceInfo info = hash.get(instance2);

			if(instance.stringValue(classIndex).equals(posClass)) {
				info.setActualClass(posClass);
			} else {
				info.setActualClass(negClass);
			}

			if(out > 0.5) {
				info.setPredictedClass(posClass);
			} else {
				info.setPredictedClass(negClass);
			}

			/*if(info.getActualClass().equals(info.getPredictedClass()) && info.getActualClass().equals(posClass)) {
				truePos ++;
			} else if(info.getActualClass().equals(info.getPredictedClass()) && info.getActualClass().equals(negClass)) {
				trueNeg ++;
			} else {
				if(info.getActualClass().equals(posClass) && info.getPredictedClass().equals(negClass)) {
					falseNeg++;
				} else {
					falsePos++;
				}
			}*/

			System.out.println((info.getStratifiedIndex()+1) + " "
					+ info.getPredictedClass() + " "
					+ info.getActualClass() + " "
					+ out);
		}

		/*System.out.println((double)trueNeg/(trueNeg + falsePos) + " " + (double)falseNeg/(trueNeg + falsePos) + " true neg, falseNeg\n" 
				+ (double)truePos/(truePos + falseNeg) + " " + (double)falsePos/(truePos + falseNeg));
*/
		/*System.out.println((double)truePredTrain/(dataInstances.numInstances()*folds) + (double)falsePredTrain/(dataInstances.numInstances()*folds) + " " 
		+(double)truePredTest/(dataInstances.numInstances()*folds) + " " + (double)falsePredTest/(dataInstances.numInstances()*folds));
		*/
	}

}
