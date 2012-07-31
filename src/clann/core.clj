(ns clann.core)

(defn zip [a,b]
 (map vector a b))

(defn sum [a]
 (reduce + a))

(defn sigmoid [v]  
 (Math/tanh (/ v 2)))

(defn multiply-pair [pair]
 (* (first pair) (second pair)))

(defn computeWeightSum [inputs, weights]
 (sum (map multiply-pair (zip inputs weights))))
