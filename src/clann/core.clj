(ns clann.core)

(use 'clojure.contrib.generic.math-functions)

(defn zip [a,b]
 (map vector a b))

(defn sum [a]
 (reduce + a))

(defn sigmoid [v]  
 (/ 1.0 (inc (exp (- v)))))

(defn inverseSigmoid [v]
  (-(log (dec (/ 1.0 v)))))

(defn multiply-pair [pair]
 (* (first pair) (second pair)))

(defn computeWeightSum [inputs, weights]
 (sum (map multiply-pair (zip inputs weights))))


(defn activation [inputs, weights]
  (sigmoid (computeWeightSum inputs weights)))

(defn makeWeights [nWeights]
  (take nWeights (repeatedly #(- 1 (rand 2)))))

(defn makeNetworkLayerWeights [nInputs, nOutputs]
  (take nOutputs (repeatedly #(makeWeights nInputs))))

(defn feedForward [inputs, layerWeights]
  (map #(activation inputs %1) layerWeights))


(defn twoLayerPerceptron [inputs, layer1Weights, layer2Weights]
  (feedForward (feedForward inputs layer1Weights) layer2Weights))


(def xorExample [[0,0],[0,1],[1,0][1,1]])
(def xorLabels (map #([apply bit-xor %1]) xorExample))

(defn diff [value1, value2]
  (- value1 value2))

(defn ones [n]
  (take n (repeatedly (fn [] 1))))

(defn neuronError [thisNeuronExpected, thisNeuronActual]
  (* thisNeuronActual 
     (- 1 thisNeuronActual) 
     (- thisNeuronExpected thisNeuronActual)))

(defn neuronWeightsDelta [thisNeuronExpected, 
                          thisNeuronActual, 
                          previousLayerActivations]
  (map #(* (neuronError thisNeuronExpected thisNeuronActual) %1) 
          previousLayerActivations))

(defn newNeuronWeights [weights, delta]
  (map #(+ (first %1) (second %1)) (zip weights delta)))


(def weights1 (makeNetworkLayerWeights 2 1))
(def sample [(sigmoid 1),(sigmoid 0)])
(def label [1])
(def layer1results (feedForward sample weights1))


(defn newWeights [e,
                  a,
                  act,
                  weights]
  (newNeuronWeights weights (neuronWeightsDelta e a act)))


(defn nextWeights [previousWeights,
                            sample,
                            label]
  (let [answer (feedForward sample previousWeights)]
    (newWeights (first label) 
                (first answer) 
                sample 
                (first previousWeights))))



(defn doEpochs [n]
  (last (take n (iterate (fn [weights] [(nextWeights weights sample label)]) weights1))))
