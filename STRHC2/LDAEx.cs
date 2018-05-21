using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RHCLib;
using Accord.Statistics;

namespace STRHC2
{
    public static class LDAEx
    {
        public const double COVARIANCE_NAN_REPLACEMENT_MUTLIPLIER = (double.Epsilon * 10) + 1;

        //public static readonly Random RND_GENERATOR = new Random(Guid.NewGuid().GetHashCode());
        // For repeatable results, I'll manually seed it for now.
        public static readonly Random RND_GENERATOR = new Random(729529);

        // For almost all methods I should do checks for rank (length), but let's assume everything passed in is correct.  This will expedite the program.
        public static bool IsCompletelySeparatedWithDiscriminant<L>(List<LabeledVector<L>> v1, List<LabeledVector<L>> v2, out DiscriminantEx<L> discriminant)
        {
            // Direct calculation: Sw^-1 * (Mean1 - Mean2)

            double[] m1 = v1.Select(v => v.Features).SpatialMean();
            double[] m2 = v2.Select(v => v.Features).SpatialMean();

            double[][] s1 = v1.Select(v => v.Features).ToArray().Covariance();
            double[][] s2 = v2.Select(v => v.Features).ToArray().Covariance();
            
            if (s1.All(r => r.All(c => double.IsNaN(c))))
            {
                foreach (double[] r in s1)
                {
                    for (int i = 0; i < r.Length; i++)
                    {
                        r[i] = LDAEx.RND_GENERATOR.Next(10) * LDAEx.COVARIANCE_NAN_REPLACEMENT_MUTLIPLIER;
                    }
                }
            }
            if (s2.All(r => r.All(c => double.IsNaN(c))))
            {
                foreach (double[] r in s2)
                {
                    for (int i = 0; i < r.Length; i++)
                    {
                        r[i] = LDAEx.RND_GENERATOR.Next(10) * LDAEx.COVARIANCE_NAN_REPLACEMENT_MUTLIPLIER;
                    }
                }
            }

            double[][] sw = s1.Add(s2);
            double[][] sw_inv = Accord.Math.Matrix.Inverse(sw);

            double[] w = Accord.Math.Matrix.Dot(sw_inv, m1.Subtract(m2));
            w = Accord.Math.Matrix.Normalize(w);    // Normalize it -- this step is ESSENTIAL

            double v1_min = double.MaxValue;
            double v1_max = double.MinValue;
            foreach (LabeledVector<L> v in v1)
            {
                double proj = Accord.Math.Matrix.Dot(w, v.Features);
                if (proj < v1_min)
                {
                    v1_min = proj;
                }
                if (proj > v1_max)
                {
                    v1_max = proj;
                }
            }

            double v2_min = double.MaxValue;
            double v2_max = double.MinValue;
            foreach (LabeledVector<L> v in v2)
            {
                double proj = Accord.Math.Matrix.Dot(w, v.Features);
                if (proj < v2_min)
                {
                    v2_min = proj;
                }
                if (proj > v2_max)
                {
                    v2_max = proj;
                }
            }

            if (v2_max < v1_min)
            {
                // Data set 1 is RIGHT of the projected SET mean
                // Data set 2 is LEFT of the projected SET mean

                double[] m = v1.Select(v => v.Features).Concat(v2.Select(v => v.Features)).SpatialMean();

                discriminant = new DiscriminantEx<L>(w, v2.First().Label, v1.First().Label, Accord.Math.Matrix.Dot(w, m), Accord.Math.Matrix.Dot(w, m2), Accord.Math.Matrix.Dot(w, m1));
                return true;
            }
            else if (v1_max < v2_min)
            {
                // Data set 1 is LEFT of the projected SET mean
                // Data set 2 is RIGHT of the projected SET mean

                double[] m = v1.Select(v => v.Features).Concat(v2.Select(v => v.Features)).SpatialMean();

                discriminant = new DiscriminantEx<L>(w, v1.First().Label, v2.First().Label, Accord.Math.Matrix.Dot(w, m), Accord.Math.Matrix.Dot(w, m1), Accord.Math.Matrix.Dot(w, m2));
                return true;
            }
            else
            {
                discriminant = null;
                return false;
            }
        }

        public static double[] SpatialMean(this IEnumerable<double[]> values)
        {
            double[] mean = new double[values.ElementAt(0).Length];

            foreach (double[] v in values)
            {
                for (int i = 0; i < v.Length; i++)
                {
                    mean[i] += v[i];
                }
            }

            int count = values.Count();
            for (int i = 0; i < mean.Length; i++)
            {
                mean[i] /= count;
            }

            return mean;
        }

        public static double[] Subtract(this double[] v1, double[] v2)
        {
            double[] result = new double[v1.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = v1[i] - v2[i];
            }

            return result;
        }

        public static double[] Add(this double[] v1, double[] v2)
        {
            double[] result = new double[v1.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = v1[i] - v2[i];
            }

            return result;
        }

        public static double[][] Add(this double[][] v1, double[][] v2)
        {
            double[][] result = new double[v1.Length][];

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = new double[v1[i].Length];

                for (int j = 0; j < v1.Length; j++)
                {
                    result[i][j] = v1[i][j] + v2[i][j];
                }
            }

            return result;
        }
    }

    public class DiscriminantEx<L>
    {
        public DiscriminantEx(double[] projectionVector, L classLeft, L classRight, double projectedSetMean, double projectedLeftMean, double projectedRightMean)
        {
            this.ProjectionVector = projectionVector;
            this.ClassLeft = classLeft;
            this.ClassRight = classRight;
            this.ProjectedSetMean = projectedSetMean;
            this.ProjectedLeftMean = projectedLeftMean;
            this.ProjectedRightMean = projectedRightMean;
        }

        public L ClassLeft { get; set; }

        public L ClassRight { get; set; }

        // Technically, it's the NORMALIZED projected left mean
        public double ProjectedLeftMean { get; set; }

        // Technically it's the NORMALIZED projected set mean
        public double ProjectedSetMean { get; set; }

        // Technically it's the NORMALIZED projected right mean
        public double ProjectedRightMean { get; set; }

        // The projection vector is already normalized
        public double[] ProjectionVector { get; set; }
    }
}
