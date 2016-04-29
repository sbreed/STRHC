//#define STRHC1

using RHCLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace STRHC2
{
    class Program
    {
        static void Main(string[] args)
        {
            System.Diagnostics.Process.GetCurrentProcess().PriorityClass = System.Diagnostics.ProcessPriorityClass.High;

            int nQueuePartitions = 1;
            int nSlotsPerQueuePartition = 4;

            if (args.Length == 2)
            {
                if (!int.TryParse(args[0], out nQueuePartitions) || !int.TryParse(args[1], out nSlotsPerQueuePartition))
                {
                    System.Console.WriteLine("Syntax: STRHC.EXE <Queue Partitions> <Slots Per Partition>");
                    return;
                }
            }

            string strFile;
            bool bIsSTRHC1 = false;

#if STRHC1
            strFile = @".\Data1.dat";
            bIsSTRHC1 = true;
#else
                strFile = @".\Data.dat";
#endif

            using (System.IO.StreamWriter sw = new System.IO.StreamWriter(string.Format(@".\{0:yyyy-MM-ddThhmmss} {1} {2}x{3}", DateTime.Now, bIsSTRHC1 ? "STRHC1" : "STRHC2", nQueuePartitions, nSlotsPerQueuePartition), false, Encoding.UTF8))
            {
                Dictionary<int, Tuple<Dictionary<string, List<List<double[]>>>, Dictionary<string, List<List<double[]>>>>> dict;

                using (System.IO.FileStream fs = new System.IO.FileStream(strFile, System.IO.FileMode.Open, System.IO.FileAccess.Read))
                {
                    BinaryFormatter bf = new BinaryFormatter();
                    dict = bf.Deserialize(fs) as Dictionary<int, Tuple<Dictionary<string, List<List<double[]>>>, Dictionary<string, List<List<double[]>>>>>;
                }

                #region Convert the dictGestures into RHCLib.LabeledVectors

                Dictionary<int, Tuple<List<RHCLib.LabeledVector<string>[]>, List<RHCLib.LabeledVector<string>[]>>> dictGestures = new Dictionary<int, Tuple<List<LabeledVector<string>[]>, List<LabeledVector<string>[]>>>();
                foreach (KeyValuePair<int, Tuple<Dictionary<string, List<List<double[]>>>, Dictionary<string, List<List<double[]>>>>> kvp in dict)
                {
                    List<LabeledVector<string>[]> lstTrain = new List<LabeledVector<string>[]>();
                    foreach (KeyValuePair<string, List<List<double[]>>> kvpTrain in kvp.Value.Item1)
                    {
                        foreach (List<double[]> gesture in kvpTrain.Value)
                        {
                            List<LabeledVector<string>> lst = new List<LabeledVector<string>>();
                            foreach (double[] frame in gesture)
                            {
                                lst.Add(new LabeledVector<string>(kvpTrain.Key, frame));
                            }
                            lstTrain.Add(lst.ToArray());
                        }
                    }

                    List<LabeledVector<string>[]> lstTest = new List<LabeledVector<string>[]>();
                    foreach (KeyValuePair<string, List<List<double[]>>> kvpTest in kvp.Value.Item2)
                    {
                        foreach (List<double[]> gesture in kvpTest.Value)
                        {
                            List<LabeledVector<string>> lst = new List<LabeledVector<string>>();
                            foreach (double[] frame in gesture)
                            {
                                lst.Add(new LabeledVector<string>(kvpTest.Key, frame));
                            }
                            lstTest.Add(lst.ToArray());
                        }
                    }

                    dictGestures.Add(kvp.Key, new Tuple<List<LabeledVector<string>[]>, List<LabeledVector<string>[]>>(lstTrain, lstTest));
                }

                #endregion

                RHCLib.DistanceDelegate measure = RHCLib.Vector.SquaredEuclideanDistance;
                IList<string> classes = dict.Values.First().Item1.Keys.ToList();
                int nClassCount = classes.Count();

                int nFeatureSpaceDimensionality = dictGestures.Values.First().Item1.First().First().Rank + (nQueuePartitions * nClassCount);

                foreach (KeyValuePair<int, Tuple<List<RHCLib.LabeledVector<string>[]>, List<RHCLib.LabeledVector<string>[]>>> kvp in dictGestures)
                {
                    RHCLib.Sphere<string> sphere = RHCLib.Sphere<string>.CreateUnitSphere(measure, nFeatureSpaceDimensionality, string.Empty);

                    int nEpoch = 1;
                    int nSpawnCount;
                    RHCLib.LabeledVector<string> lvectorAug;

                    Program.WriteStream(sw, string.Format("Fold: {0}", kvp.Key));

                    System.Diagnostics.Stopwatch watch = new System.Diagnostics.Stopwatch();
                    watch.Start();

                    #region Train

                    do
                    {
                        nSpawnCount = 0;

                        foreach (RHCLib.LabeledVector<string>[] gesture in kvp.Value.Item1)
                        {
                            double[][] queue = Program.CreateEmptyQueue(nQueuePartitions, nSlotsPerQueuePartition, nClassCount);

                            foreach (RHCLib.LabeledVector<string> frame in gesture)
                            {
                                lvectorAug = Program.CreateAugmentedVector(frame, queue, nQueuePartitions, nSlotsPerQueuePartition);

                                double[] biases = sphere.CalculateBiases(lvectorAug, measure, classes);

                                nSpawnCount += sphere.SpawnSingular(lvectorAug, measure, true).Count();

                                Program.AdvanceQueue(queue, biases);
                            }
                        }

                        Program.WriteStream(sw, string.Format("  Epoch:\t{0}\t{1}", nEpoch++, nSpawnCount));
                    } while (nSpawnCount > 0);

                    watch.Stop();

                    Program.WriteStream(sw, string.Format("TOTAL TIME [ms] for Fold {0}: {1}", kvp.Key, watch.ElapsedMilliseconds.ToString()));
                    Program.WriteStream(sw, string.Format("Sphere Count: {0}", sphere.SphereCount));
                    Program.WriteStream(sw, string.Format("Tree Height: {0}", sphere.Height));
                    Program.WriteStream(sw, string.Format("Epoch Count: {0}", nEpoch - 1));

                    #endregion

                    #region Test

                    Sphere<string> sphereWinner = null;

                    int nSphereCorrect = 0;
                    int nSphereIncorrect = 0;
                    int nQueueCorrect = 0;
                    int nQueueIncorrect = 0;

                    foreach (RHCLib.LabeledVector<string>[] gesture in kvp.Value.Item2)
                    {
                        #region Create Empty Queue

                        double[][] queue = new double[nQueuePartitions * nSlotsPerQueuePartition][];
                        for (int i = 0; i < queue.Length; i++)
                        {
                            queue[i] = new double[nClassCount];
                        }

                        #endregion

                        watch.Reset();
                        watch.Start();

                        foreach (RHCLib.LabeledVector<string> frame in gesture)
                        {
                            lvectorAug = Program.CreateAugmentedVector(frame, queue, nQueuePartitions, nSlotsPerQueuePartition);

                            sphereWinner = sphere.Recognize(lvectorAug, measure);

                            double[] rgBiases = sphere.CalculateBiases(lvectorAug, measure, classes);

                            Program.AdvanceQueue(queue, rgBiases);
                        }
                        watch.Stop();

                        Program.WriteStream(sw, string.Format("[{0} fold] Time to recognize [in ms]: {1}", kvp.Key, watch.ElapsedMilliseconds));

                        #region Sphere Winner

                        Program.WriteStream(sw, string.Format("Actual: {0}\t\tSphere Recognized: {1}", gesture.First().Label, sphereWinner.Label));
                        if (sphereWinner.Label == gesture.First().Label)
                        {
                            nSphereCorrect++;
                        }
                        else
                        {
                            nSphereIncorrect++;
                        }

                        #endregion

                        #region Queue Winner

                        Dictionary<string, double> dictWinner = new Dictionary<string, double>();
                        foreach (string strClass in classes)
                        {
                            dictWinner.Add(strClass, 0.0);
                        }

                        // Quick and dirty
                        Dictionary<string, double> dictQueueBreakout = new Dictionary<string, double>();
                        foreach (string strClass in classes)
                        {
                            dictQueueBreakout.Add(strClass, 0.0);
                        }
                        for (int i = 0; i < queue.Length; i++)
                        {
                            for (int j = 0; j < queue[i].Length; j++)
                            {
                                dictQueueBreakout[classes.ElementAt(j)] += queue[i][j];
                            }
                        }

                        string strQueueWinner = dictQueueBreakout.Aggregate((l, r) => l.Value > r.Value ? l : r).Key;

                        Program.WriteStream(sw, string.Format("Actual: {0}\t\tQueue Recognized: {1}", gesture.First().Label, strQueueWinner));
                        if (strQueueWinner == gesture.First().Label)
                        {
                            nQueueCorrect++;
                        }
                        else
                        {
                            nQueueIncorrect++;
                        }

                        #endregion
                    }

                    Program.WriteStream(sw, string.Format("Sphere Correct: {0}\t\tSphere Incorrect: {1}\t\tSphere Percentage: {2:P}", nSphereCorrect, nSphereIncorrect, (double)nSphereCorrect / (nSphereCorrect + nSphereIncorrect)));
                    Program.WriteStream(sw, string.Format("Queue Correct: {0}\t\tQueue Incorrect: {1}\t\tQueue Percentage: {2:P}", nQueueCorrect, nQueueIncorrect, (double)nQueueCorrect / (nQueueCorrect + nQueueIncorrect)));

                    Program.WriteStream(sw, System.Environment.NewLine);
                    Program.WriteStream(sw, System.Environment.NewLine);

                    #endregion

                    #region Serialize Sphere

                    using (System.IO.FileStream fs = new System.IO.FileStream(string.Format(@".\{0}-{1}-{2}-{3}.serialized", bIsSTRHC1 ? "STRHC1" : "STRHC2", nQueuePartitions, nSlotsPerQueuePartition, kvp.Key), System.IO.FileMode.Create))
                    {
                        BinaryFormatter bf = new BinaryFormatter();
                        bf.Serialize(fs, sphere);
                    }

                    #endregion
                }
            }
        }

        private static void AdvanceQueue(double[][] queue, double[] biases)
        {
            for (int i = queue.Length - 1; i > 0; i--)
            {
                queue[i] = queue[i - 1];
            }

            queue[0] = biases;
        }

        private static LabeledVector<string> CreateAugmentedVector(LabeledVector<string> frame, double[][] queue, int nQueuePartitions, int nSlotsPerQueuePartition)
        {
            int nClassCount = queue[0].Length;
            double[] rgAppend = new double[nQueuePartitions * nClassCount];

            for (int i = 0; i < queue.Length; i++)
            {
                for (int j = 0; j < queue[i].Length; j++)
                {
                    rgAppend[((i / nSlotsPerQueuePartition) * nClassCount) + j] += queue[i][j];
                }
            }

            #region Normalize from 0.0 to 1.0
            // Need to normalize the PARTITIONS and not the whole queue

            double fSum = 0.0;
            for (int i = 0; i < rgAppend.Length; i++)
            {
                fSum += rgAppend[i];

                if ((i > 0 && i % nClassCount == nClassCount - 1))
                {
                    for (int j = i - (nClassCount - 1); j <= i; j++)
                    {
                        rgAppend[j] = (fSum > 0.0) ? (rgAppend[j] / fSum) : 0.0;
                    }

                    fSum = 0.0;
                }
            }

            #endregion

            return new LabeledVector<string>(frame.Label, frame.Features.Concat(rgAppend));
        }

        private static double[][] CreateEmptyQueue(int nQueuePartitions, int nSlotsPerQueuePartition, int nClassCount)
        {
            double[][] queue = new double[nQueuePartitions * nSlotsPerQueuePartition][];
            for (int i = 0; i < queue.Length; i++)
            {
                queue[i] = new double[nClassCount];
            }

            return queue;
        }

        public static void WriteStream(System.IO.StreamWriter sw, string value)
        {
            sw.WriteLine(value);
            sw.Flush(); // Flush the stream, so I can open it in Notepad and view progress even though the stream is still open

            // Mirror
            System.Diagnostics.Debug.WriteLine(value);
        }
    }

    public static class ExtensionMethods
    {
        public static double[] CalculateBiases<L>(this Sphere<L> sphere, Vector vector, RHCLib.DistanceDelegate measure, IList<L> labels)
        {
            SortedDictionary<L, double?> dictClasses = new SortedDictionary<L, double?>();
            foreach (L label in labels)
            {
                dictClasses.Add(label, null);
            }

            double fProportion = 1.0;
            int nCount = dictClasses.Count;
            bool bFirstSphere = true;

            RHCLib.Sphere<L> sphereIteration = sphere.Recognize(vector, measure, true);
            while (sphereIteration != null && fProportion > 0.0 && nCount > 0)
            {
                if (bFirstSphere)
                {
                    if (dictClasses.ContainsKey(sphereIteration.Label))
                    {
                        #region Linear

                        dictClasses[sphereIteration.Label] = fProportion - (measure(sphereIteration, vector) * (0.5 / sphereIteration.Radius));

                        #endregion

                        fProportion -= dictClasses[sphereIteration.Label].Value;

                        bFirstSphere = false;
                        nCount--;
                    }
                }
                else
                {
                    if (dictClasses.ContainsKey(sphereIteration.Label) && !dictClasses[sphereIteration.Label].HasValue)
                    {
                        dictClasses[sphereIteration.Label] = fProportion;
                        nCount--;
                    }

                    #region Linear

                    fProportion -= fProportion * ((sphereIteration.Radius - measure(sphereIteration, vector)) / sphereIteration.Radius);

                    #endregion
                }

                sphereIteration = sphereIteration.Parent;
            }

            double fSum = dictClasses.Values.Sum(v => v.HasValue ? v.Value : 0.0);

            return dictClasses.Values.Select(v => v.HasValue ? v.Value / fSum : 0.0).ToArray();
        }
    }
}
