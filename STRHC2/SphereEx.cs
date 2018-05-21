using RHCLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace STRHC2
{
    [Serializable]
    public class SphereEx<L> : Sphere<L>
    {
        public SphereEx(double fRadius, LabeledVector<L> lvector) : base(fRadius, lvector)
        {

        }

        public SphereEx(double fRadius, L label, IEnumerable<double> features) : base(fRadius, label, features)
        {

        }

        public SphereEx(double fRadius, L label, params double[] features) : base(fRadius, label, features)
        {

        }

        public SphereEx(double fRadius, L label, IVector vector) : base(fRadius, label, vector)
        {

        }

        public SphereEx(Sphere<L> sphere): this(sphere.Radius, sphere.Label, (IVector)sphere)
        {

        }

        public void Cleanup()
        {
            foreach (SphereEx<L> child in this.Children.OfType<SphereEx<L>>())
            {
                child.Cleanup();
            }

            this.LDAVectors = null;
        }

        public override Sphere<L> Recognize(IVector vector, DistanceDelegate measure, ParallelStrategy parallelStrategy)
        {
            if (this is SphereEx<L> && ((SphereEx<L>)this).DiscriminantEx != null)
            {
                Sphere<L> sphereMinRadius = this.Parent == null ? this : null;

                if (this.DoesEncloseVector(vector, measure))
                {
                    double proj = Accord.Math.Matrix.Dot(this.DiscriminantEx.ProjectionVector, vector.Features);
                    if (proj <= this.DiscriminantEx.ProjectedSetMean)
                    {
                        // Should be the ClassLabelLeft label

                        // Graphical Example:
                        //
                        //      |-------X--------+----------------|       X => Decision point calculated by LDA,     + => Centroid of Sphere
                        //      |-------|                                 <-- If a point falls left of the decision point, I need the segment between | and X.  Remember to divide by 2.0 because it should be a radius.
                        //          ^
                        //          |
                        //      Remember to divide by two as it's a radius

                        double farLeft = Accord.Math.Matrix.Dot(this.DiscriminantEx.ProjectionVector, this.Features) - (this.Radius * this.Radius);
                        if (farLeft > this.DiscriminantEx.ProjectedLeftMean)
                        {
                            // Squared Euclidean may have a problem because the ProjectLeftMean may be left of the hypersphere
                            // As such, add a small buffer (distance between farLeft and ProjectedLeftMean)
                            // This is, perhaps, not the best method, but it works.  Maybe explore other solutions when using Squared Euclidean Distance with LDA.

                            sphereMinRadius = new Sphere<L>((this.DiscriminantEx.ProjectedSetMean - this.DiscriminantEx.ProjectedLeftMean) / 2.0, this.DiscriminantEx.ClassLeft, vector);
                        }
                        else
                        {
                            sphereMinRadius = new Sphere<L>((this.DiscriminantEx.ProjectedSetMean - farLeft) / 2.0, this.DiscriminantEx.ClassLeft, vector);
                        }
                    }
                    else
                    {
                        // Should be the ClassLabelRight label

                        double farRight = Accord.Math.Matrix.Dot(this.DiscriminantEx.ProjectionVector, this.Features) + (this.Radius * this.Radius);

                        if (farRight < this.DiscriminantEx.ProjectedRightMean)
                        {
                            // Squared Euclidean may have a problem because the ProjectRightMean may be right of the hypersphere
                            // As such, add a small buffer (distance between farRight and ProjectedRightMean)
                            // This is, perhaps, not the best method, but it works.  Maybe explore other solutions when using Squared Euclidean Distance with LDA.

                            sphereMinRadius = new Sphere<L>((this.DiscriminantEx.ProjectedRightMean - this.DiscriminantEx.ProjectedSetMean) / 2.0, this.DiscriminantEx.ClassRight, vector);
                        }
                        else
                        {
                            sphereMinRadius = new Sphere<L>((farRight - this.DiscriminantEx.ProjectedSetMean) / 2.0, this.DiscriminantEx.ClassRight, vector);
                        }
                    }
                }

                return sphereMinRadius;
            }
            else
            {
                return base.Recognize(vector, measure, parallelStrategy);
            }
        }

        public override L RecognizeAsLabel(IVector vector, DistanceDelegate measure, ParallelStrategy parallelStrategy)
        {
            Sphere<L> winner = this.Recognize(vector, measure, parallelStrategy);
            return winner != null ? winner.Label : default(L);
        }

        public int Spawn(LabeledVector<L> lvector, DistanceDelegate measure, ParallelStrategy parallelStrategy, bool spawnUsingLDA)
        {
            int nSpawnCount = 0;
            if (this.DoesEncloseVector(lvector, measure))
            {
                if (parallelStrategy == ParallelStrategy.Multithreaded)
                {
                    List<Task> lstSpawnThreads = new List<Task>();
                    foreach (SphereEx<L> child in this.Children)
                    {
                        lstSpawnThreads.Add(Task.Factory.StartNew(c =>
                        {
                            nSpawnCount += ((SphereEx<L>)c).Spawn(lvector, measure, ParallelStrategy.SingleThreaded, spawnUsingLDA);
                        }, child, TaskCreationOptions.LongRunning));
                    }

                    Task.WaitAll(lstSpawnThreads.ToArray());
                }
                else
                {
                    foreach (SphereEx<L> child in this.Children)
                    {
                        nSpawnCount += child.Spawn(lvector, measure, ParallelStrategy.SingleThreaded, spawnUsingLDA);
                    }
                }

                if (!spawnUsingLDA)
                {
                    #region Regular Spawn

                    if (!this.Label.Equals(lvector.Label) && !Vector.EqualsEx(this, lvector) && !this.DoesAtLeastOneChildEncloseVector(lvector, measure))
                    {
                        SphereEx<L> child = new SphereEx<L>(this.Radius - measure(this, lvector), lvector);
                        child.LDAVectors.Add(child.Label, new List<KeyValuePair<int, LabeledVector<L>>>() { new KeyValuePair<int, LabeledVector<L>>(0, lvector) });
                        this.AddChild(child);
                        nSpawnCount++;
                    }

                    #endregion
                }
                else
                {
                    #region LDA Spawn

                    // Note that LDA Spawn doesn't work with SquaredEuclideanDistance.  I may -- in the future -- do an additional check to see if the vector is contained by the hypersphere with SquaredEuclideanDistance and EuclideanDistance.  If so, this will work.  Until then, I'll leave it as is.
                    if (this.Children.Any())
                    {
                        // Next if-statement a duplicate of the region "Regular Spawn"
                        // Remember, LDA is only applied at nodes with no children (leaf)
                        if (!this.Label.Equals(lvector.Label) && !Vector.EqualsEx(this, lvector) && !this.DoesAtLeastOneChildEncloseVector(lvector, measure))
                        {
                            SphereEx<L> child = new SphereEx<L>(this.Radius - measure(this, lvector), lvector);
                            child.LDAVectors.Add(child.Label, new List<KeyValuePair<int, LabeledVector<L>>>() { new KeyValuePair<int, LabeledVector<L>>(0, lvector) });
                            this.AddChild(child);
                            nSpawnCount++;
                        }
                    }
                    else
                    {
                        bool bContains = false;

                        List<KeyValuePair<int, LabeledVector<L>>> lst;
                        if (!this.LDAVectors.TryGetValue(lvector.Label, out lst))
                        {
                            this.LDAVectors.Add(lvector.Label, (lst = new List<KeyValuePair<int, LabeledVector<L>>>()));
                        }
                        else
                        {
                            bContains = lst.Any(kvp => Vector.EqualsEx(kvp.Value, lvector));
                        }

                        if (!bContains)
                        {
                            lst.Add(new KeyValuePair<int, LabeledVector<L>>(this.LDAVectors.Sum(kvp => kvp.Value.Count), lvector));

                            if (this.LDAVectors.Keys.Count == 2)
                            {
                                #region If the sphere contains exactly 2 classes, do the following...

                                bool bIsSeparable = false;
                                DiscriminantEx<L> discriminant = null;
                                try
                                {
                                    bIsSeparable = LDAEx.IsCompletelySeparatedWithDiscriminant(this.LDAVectors.ElementAt(0).Value.Select(kvp => kvp.Value).ToList(), this.LDAVectors.ElementAt(1).Value.Select(kvp => kvp.Value).ToList(), out discriminant);
                                }
                                catch
                                {
                                    // Just consume, leaving bIsSeparable = false
                                }

                                if (!bIsSeparable)
                                {
                                    bool spawnedAtLeastOnce = false;
                                    List<LabeledVector<L>> lstCache = this.LDAVectors.SelectMany(kvp => kvp.Value).OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToList();

                                    this.LDAVectors.Clear();
                                    this.DiscriminantEx = null;

                                    foreach (LabeledVector<L> v in lstCache)
                                    {
                                        if (!spawnedAtLeastOnce)
                                        {
                                            int count = this.Spawn(v, measure, ParallelStrategy.SingleThreaded, false);
                                            nSpawnCount += count;
                                            spawnedAtLeastOnce = count > 0;
                                        }
                                        else
                                        {
                                            nSpawnCount += this.Spawn(v, measure, ParallelStrategy.SingleThreaded, true);
                                        }
                                    }
                                }
                                else
                                {
                                    if (discriminant != null)
                                    {
                                        this.DiscriminantEx = discriminant;
                                    }
                                }

                                #endregion
                            }
                            else if (this.LDAVectors.Keys.Count > 2)
                            {
                                #region If the sphere contains 3 or more classes, do the following...

                                bool spawnedAtLeastOnce = false;
                                List<LabeledVector<L>> lstCache = this.LDAVectors.SelectMany(kvp => kvp.Value).OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToList();

                                this.LDAVectors.Clear();
                                this.DiscriminantEx = null;

                                foreach (LabeledVector<L> v in lstCache)
                                {
                                    if (!spawnedAtLeastOnce)
                                    {
                                        int count = this.Spawn(v, measure, ParallelStrategy.SingleThreaded, false);
                                        nSpawnCount += count;
                                        spawnedAtLeastOnce = count > 0;
                                    }
                                    else
                                    {
                                        nSpawnCount += this.Spawn(v, measure, ParallelStrategy.SingleThreaded, true);
                                    }
                                }

                                #endregion
                            }
                            // Note: If this.LDAVectors.Keys.Count == 1, don't do anything additional.
                        }
                    }

                    #endregion
                }
            }

            return nSpawnCount;
        }

        private Dictionary<L, List<KeyValuePair<int, LabeledVector<L>>>> LDAVectors { get; set; } = new Dictionary<L, List<KeyValuePair<int, LabeledVector<L>>>>();

        public int CountOfLDASpheres
        {
            get
            {
                int count = 0;
                foreach (SphereEx<L> child in this.Children.OfType<SphereEx<L>>())
                {
                    count += child.CountOfLDASpheres;
                }

                if (this.DiscriminantEx != null)
                {
                    count++;
                }

                return count;
            }
        }

        public DiscriminantEx<L> DiscriminantEx { get; set; }

        private void OldLDASpawnRegion()
        {
            #region LDA Spawn

                    //if (!this.DoesAtLeastOneChildEncloseVector(lvector, measure))   // Don't care about label as well as location of lvector
                    //{
                    //    bool bContains = false;

                    //    List<LabeledVector<L>> lst;
                    //    if (!this.LDAVectors.TryGetValue(lvector.Label, out lst))
                    //    {
                    //        this.LDAVectors.Add(lvector.Label, (lst = new List<LabeledVector<L>>()));
                    //        lst.Add(lvector);
                    //    }
                    //    else
                    //    {
                    //        bContains = lst.Any(v => Vector.EqualsEx(v, lvector));
                    //    }

                    //    if (!bContains)
                    //    {
                    //        if (this.LDAVectors.Keys.Count == 2)
                    //        {
                    //            #region If the sphere contains exactly 2 classes, do the following...
                    //            // 1.  Create another discriminant
                    //            // 2.  If the discrimant CANNOT separate the classes, do the following...
                    //            // 2a.    Remove the lvector from the list it was just added to
                    //            // 2b.    Create new children from the classes
                    //            // 2c.    Try to spawn the presented LVector in the NEW children
                    //            // 2d.    Clear the dictionary, LDAVectors
                    //            // 2e.    If you can't spawn WITHIN the NEW children, add the vector to the this.LDAVectors
                    //            // 3.  Else (If the discrimant CAN separate the classes), do nothing but assign the property.

                    //            Discriminant<L> discriminant = null;
                    //            bool bIsSeparable = false;
                    //            bool bIsExceptionThrown = false;
                    //            try
                    //            {
                    //                bIsSeparable = LDA.IsCompletelySeparatedWithDiscriminant(this.LDAVectors.ElementAt(0).Value, this.LDAVectors.ElementAt(1).Value, this, out discriminant);
                    //            }
                    //            catch
                    //            {
                    //                bIsExceptionThrown = true;
                    //                // Just consume, leaving bIsSeparable = false
                    //            }

                    //            if (!bIsSeparable)
                    //            {
                    //                lst.RemoveAt(lst.Count - 1);    // Faster than .Remove() as I am not linearly searching

                    //                List<SphereEx<L>> lstNewChildren = new List<SphereEx<L>>();
                    //                foreach (KeyValuePair<L, List<LabeledVector<L>>> kvp in this.LDAVectors)
                    //                {
                    //                    if (kvp.Value.Any())
                    //                    {
                    //                        Vector vectorCentroid = Vector.Centroid(kvp.Value);

                    //                        if (!Vector.EqualsEx(this, vectorCentroid))
                    //                        {
                    //                            SphereEx<L> child = new SphereEx<L>(this.Radius - measure(this, vectorCentroid), kvp.Key, (IVector)vectorCentroid);

                    //                            this.AddChild(child);
                    //                            lstNewChildren.Add(child);
                    //                            nSpawnCount++;
                    //                        }
                    //                    }
                    //                }

                    //                bool hasSpawned = false;
                    //                foreach (SphereEx<L> child in lstNewChildren)
                    //                {
                    //                    if (child.DoesEncloseVector(lvector, measure))
                    //                    {
                    //                        nSpawnCount += child.Spawn(lvector, measure, ParallelStrategy.SingleThreaded, spawnUsingLDA);
                    //                        hasSpawned = true;
                    //                    }
                    //                }

                    //                this.LDAVectors.Clear();

                    //                if (!hasSpawned)
                    //                {
                    //                    this.LDAVectors.Add(lvector.Label, new List<LabeledVector<L>>() { lvector });
                    //                }
                    //            }
                    //            else if (!bIsExceptionThrown)
                    //            {
                    //                this.Discriminant = discriminant;
                    //            }

                    //            #endregion
                    //        }
                    //        else if (this.LDAVectors.Keys.Count > 2)
                    //        {
                    //            #region If the sphere contains 3 or more classes, do the following...
                    //            // 1.  Create children from the OLDER label-sets
                    //            // 2.  Try to spawn the presented LVector IN the NEW children just created
                    //            // 3.  Clear the dictionary, LDAVectors
                    //            // 4.  If you can't spawn WITHIN the NEW children, add the vector to this.LDAVectors

                    //            List<SphereEx<L>> lstNewChildren = new List<SphereEx<L>>();

                    //            foreach (KeyValuePair<L, List<LabeledVector<L>>> kvp in this.LDAVectors)
                    //            {
                    //                if (!object.Equals(kvp.Key, lvector.Label))
                    //                {
                    //                    Vector vectorCentroid = Vector.Centroid(kvp.Value);

                    //                    if (!Vector.EqualsEx(this, vectorCentroid))
                    //                    {
                    //                        SphereEx<L> child = new SphereEx<L>(this.Radius - measure(this, vectorCentroid), kvp.Key, (IVector)vectorCentroid);

                    //                        this.AddChild(child);
                    //                        lstNewChildren.Add(child);
                    //                        nSpawnCount++;
                    //                    }
                    //                }
                    //            }

                    //            bool hasSpawned = false;
                    //            foreach (SphereEx<L> child in lstNewChildren)
                    //            {
                    //                if (child.DoesEncloseVector(lvector, measure))
                    //                {
                    //                    nSpawnCount += child.Spawn(lvector, measure, ParallelStrategy.SingleThreaded, spawnUsingLDA);
                    //                    hasSpawned = true;
                    //                }
                    //            }

                    //            this.LDAVectors.Clear();

                    //            if (!hasSpawned)
                    //            {
                    //                this.LDAVectors.Add(lvector.Label, new List<LabeledVector<L>>() { lvector });
                    //            }

                    //            #endregion
                    //        }
                    //        // Note: If this.LDAVectors.Keys.Count == 1, don't do anything additional.
                    //    }
                    //}

                    #endregion
        }
    }
}