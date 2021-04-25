import scalismo.statisticalmodel.PointDistributionModel
import scalismo.geometry._
import scalismo.common._
import scalismo.mesh._
import scalismo.statisticalmodel.MultivariateNormalDistribution
import scalismo.numerics.UniformMeshSampler3D
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.ui.api._
import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.statisticalmodel.dataset.DataCollection

object FindCorrespondence {

    def main(args: Array[String]) : Unit = {
      implicit val rng = scalismo.utils.Random(42L)
      val ui = ScalismoUI()

      for (i <- 0 until 47)
      {
        val targetMesh = MeshIO.readMesh(new java.io.File(s"datasets/challenge-data/challengedata/aligned-full-femurs/meshes/$i.stl")).get
        val model = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new java.io.File("datasets/challenge-data/challengedata/GaussianProcessModel/GaussianProcessModel.h5")).get

        //Selects the points for which we want to find the correspondences - uniformly distributed on the surface
        val sampler = UniformMeshSampler3D(model.reference, numberOfPoints = 5000)
        val points: Seq[Point[_3D]] = sampler.sample.map(pointWithProbability => pointWithProbability._1)

        //Uses point ids of the sampled points
        val ptIds = points.map(point => model.reference.pointSet.findClosestPoint(point).id)

        //Finds for each point of interest the closest point on the target
        def attributeCorrespondences(movingMesh: TriangleMesh[_3D], ptIds: Seq[PointId]): Seq[(PointId, Point[_3D])] = {
          ptIds.map { id: PointId =>
            val pt = movingMesh.pointSet.point(id)
            val closestPointOnMesh2 = targetMesh.pointSet.findClosestPoint(pt).point
            (id, closestPointOnMesh2)
          }
        }

        val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3))

        //Computes a Gaussian process regression using the correspondences
        def fitModel(correspondences: Seq[(PointId, Point[_3D])]): TriangleMesh[_3D] = {
          val regressionData = correspondences.map(correspondence =>
            (correspondence._1, correspondence._2, littleNoise)
          )
          val posterior = model.posterior(regressionData.toIndexedSeq)
          posterior.mean
        }

        //Iterates the procedure
        def nonrigidICP(movingMesh: TriangleMesh[_3D], ptIds: Seq[PointId], numberOfIterations: Int): TriangleMesh[_3D] = {
          if (numberOfIterations == 0) movingMesh
          else {
            val correspondences = attributeCorrespondences(movingMesh, ptIds)
            val transformed = fitModel(correspondences)
            nonrigidICP(transformed, ptIds, numberOfIterations - 1)
          }
        }
        //Repeats the fitting steps iteratively for 20 times
        val finalFit = nonrigidICP(model.mean, ptIds, 20)

        //Stores the model
        MeshIO.writeMesh(finalFit, new java.io.File(s"datasets/challenge-data/challengedata/coresponded-full-femurs/meshes/$i.stl")).get
      }

      val meshFiles = new java.io.File("datasets/challenge-data/challengedata/coresponded-full-femurs/meshes/").listFiles
      val (meshes) = meshFiles.map(meshFile => {
        val mesh = MeshIO.readMesh(meshFile).get
        (mesh)
      })
      val meshes2 : Array[TriangleMesh[_3D]] = meshes
      val reference = MeshIO.readMesh(new java.io.File("data/femur.stl")).get

      //Learns a PCA model from all the meshes in correspondences
      val dc = DataCollection.fromTriangleMesh3DSequence(reference, meshes2)
      val modelFromDataCollection = PointDistributionModel.createUsingPCA(dc)

      val modelGroup = ui.createGroup("modelGroup2")
      ui.show(modelGroup, modelFromDataCollection, "PCAModel")
      //Stores the PCA model
      StatisticalModelIO.writeStatisticalTriangleMeshModel3D(modelFromDataCollection, new java.io.File("datasets/challenge-data/challengedata/GaussianProcessModel/PCAModel.h5"))
    }
}