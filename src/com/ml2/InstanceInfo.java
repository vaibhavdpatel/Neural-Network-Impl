package com.ml2;

public class InstanceInfo {
	int actualIndex, stratifiedIndex;
	String actualClass, predictedClass;

	public String getActualClass() {
		return actualClass;
	}

	public void setActualClass(String actualClass) {
		this.actualClass = actualClass;
	}

	public String getPredictedClass() {
		return predictedClass;
	}

	public void setPredictedClass(String predictedClass) {
		this.predictedClass = predictedClass;
	}

	public int getActualIndex() {
		return actualIndex;
	}

	public void setActualIndex(int actualIndex) {
		this.actualIndex = actualIndex;
	}

	public int getStratifiedIndex() {
		return stratifiedIndex;
	}

	public void setStratifiedIndex(int stratifiedIndex) {
		this.stratifiedIndex = stratifiedIndex;
	}
	
}
