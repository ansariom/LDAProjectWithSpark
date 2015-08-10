/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.examples.mllib;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.rdd.RDD;

import scala.Tuple2;

public class JavaLDAExample {
  public static void main(String[] args) {
    SparkConf conf = new SparkConf().setMaster("local").setAppName("My app");
    JavaSparkContext sc = new JavaSparkContext(conf);

    // Load and parse the data
    String path = "data/mllib/sample_lda_data.txt";
    JavaRDD<String> data = sc.textFile(path);
    JavaRDD<Vector> parsedData = data.map(
        new Function<String, Vector>() {
          public Vector call(String s) {
            String[] sarray = s.trim().split(" ");
            double[] values = new double[sarray.length];
            for (int i = 0; i < sarray.length; i++)
              values[i] = Double.parseDouble(sarray[i]);
            return Vectors.dense(values);
          }
        }
    );
    // Index documents with unique IDs
    JavaPairRDD<Long, Vector> corpus = JavaPairRDD.fromJavaRDD(parsedData.zipWithIndex().map(
        new Function<Tuple2<Vector, Long>, Tuple2<Long, Vector>>() {
          public Tuple2<Long, Vector> call(Tuple2<Vector, Long> doc_id) {
            return doc_id.swap();
          }
        }
    ));
    corpus.cache();

    // Cluster the documents into three topics using LDA
    DistributedLDAModel ldaModel = (DistributedLDAModel)new LDA().setK(3).run(corpus);

    // Output topics. Each is a distribution over words (matching word count vectors)
    System.out.println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize()
        + " words):");
    
    JavaPairRDD<Long, Vector> docDistJava = ldaModel.javaTopicDistributions();
    Map<Long, Vector> map = docDistJava.collectAsMap();
    Iterator<Long> keyset = map.keySet().iterator();
    while (keyset.hasNext()) {
		Long docId = (Long) keyset.next();
		System.out.println("**** Document " + docId);
		for (int i = 0; i < 3; i++) {
			System.out.print("\t\t" + map.get(docId).apply(i));
		}
		System.out.println();
	}
    
    
//    for (int i = 0; i < 6; i++) {
//    	System.out.println("Doc****** " + i + " : " + docDistJava.lookup((long) i).get(0));
//    	System.out.println("Doc " + i + " : " + docDistJava.lookup((long) i).get(1));
//   		System.out.println("Doc " + i + " : " + docDistJava.lookup((long) i).get(2));
//	}
//    RDD<scala.Tuple2<Object, Vector>> docDist = ldaModel.topicDistributions();
    
    
    Matrix topics = ldaModel.topicsMatrix();
    for (int topic = 0; topic < 3; topic++) {
      System.out.print("Topic " + topic + ":");
      for (int word = 0; word < ldaModel.vocabSize(); word++) {
        System.out.print(" " + topics.apply(word, topic));
      }
      System.out.println();
    }
    sc.stop();
  }
}
