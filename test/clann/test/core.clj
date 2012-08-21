(ns clann.test.core
  (:use [clann.core])
  (:use [clojure.test]))

(def weights1 (makeNetworkLayerWeights 2 1))
(def sample [(sigmoid 1),(sigmoid 0)])
(def label [1])
(def layer1results (feedForward sample weights1))

(defn round [v]
  (let [n (int v)]
    (if (>= (- v n) 0.5)
      (inc n)
      n)))

(deftest sigmoid-inverse-sigmoid
         (is (> 0.000001 (Math/abs (- 0.5 (inverseSigmoid (sigmoid 0.5))))))
         (is (> 0.000001 (let [r (rand)] (- r (inverseSigmoid (sigmoid r)))))))

(deftest test-sum
 (is (= 7 (sum '(1,2,4)))))

(deftest weight-activation
 (is (= 14 (computeWeightSum '(1,2,4) '(9,2,0.25)))))

(deftest activation-test
         (is (= 1 (activation '(100) '(100))))
         (is (= 0.5 (activation '(0) '(0))))
         (is (< 0.5 (activation '(1) '(2))))
         (is (< 0.5 (activation '(1) '(0.3)))))

(deftest make-weights-test
         (is (= 10 (count (makeWeights 10)))))

(deftest make-network-layer-weights-test
         (is (= 3 (count (makeNetworkLayerWeights 1 3))))
         (is (= 1 (count (first  (makeNetworkLayerWeights 1 3)  ))))
         (is (= 1 (count (second (makeNetworkLayerWeights 1 3)  ))))
         (is (= 1 (count (nth    (makeNetworkLayerWeights 1 3) 2)))))

(deftest feed-forward-test
         (is (= [1,1] (feedForward [1,2] [[100,0],[0,50]])))
         (is (= [0.5, 0.5] (feedForward [1,2] [[0,0],[0,0]]))))

(deftest propagate-simple-test
         (is (> 0.99
                (first
                  (feedForward
                    sample (doEpochs 1000 weights1 sample label))))))


(deftest propogate-complex-test
         (let [result (feedForward [1,0] 
                                   (doEpochs 3000 
                                             (makeNetworkLayerWeights 2 3) 
                                             [1,0] 
                                             [1,0,1]))]
           (is (== (round (first result))  1))
           (is (== (round (second result)) 0))
           (is (== (round (nth result 2))  1))))

(deftest two-layer-test
         (let [hiddenWeights (makeNetworkLayerWeights 2 3)
               outputWeights (makeNetworkLayerWeights 3 1)
               trainedWeights (trainTwoLayerPerceptron
                                3000
                                [hiddenWeights, outputWeights]
                                sample
                                [1])
               result (twoLayerPerceptron sample (first trainedWeights) (second trainedWeights))]
           (is (== (round (first result)) 1))
           (is (== (count result) 1))))

(deftest complex-two-layer-test
         (let [hiddenWeights (makeNetworkLayerWeights 2 8)
               outputWeights (makeNetworkLayerWeights 8 3)
               trainedWeights (trainTwoLayerPerceptron
                                3000
                                [hiddenWeights, outputWeights]
                                sample
                                [1,0,1])
               result (twoLayerPerceptron sample (first trainedWeights) (second trainedWeights))]
           (is (== (round (first result))  1))
           (is (== (round (second result)) 0))
           (is (== (round (nth result 2))  1))
           (is (== (count result) 3))))
