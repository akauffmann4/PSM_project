import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel3D}
import scalismo.statisticalmodel.{GaussianProcess3D, LowRankGaussianProcess, PointDistributionModel}
import scalismo.geometry._
import scalismo.common._
import scalismo.mesh._
import scalismo.statisticalmodel.MultivariateNormalDistribution
import scalismo.numerics.UniformMeshSampler3D
import scalismo.io.{LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.ui.api._
import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.statisticalmodel.dataset.DataCollection
//import scalismo.registration._

object FindCorrespondence {

    def main(args: Array[String]) : Unit = {
      implicit val rng = scalismo.utils.Random(42L)
      val ui = ScalismoUI()

      var i = 10;
      for (i <- 0 until 47)
      {
        System.out.println(i)
        val targetMesh = MeshIO.readMesh(new java.io.File(s"datasets/challenge-data/challengedata/aligned-full-femurs/meshes/$i.stl")).get
        val model = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new java.io.File("datasets/challenge-data/challengedata/GaussianProcessModel/GaussianProcessModel.h5")).get

        //val targetGroup = ui.createGroup("targetGroup")
        //val targetMeshView = ui.show(targetGroup, targetMesh, "targetMesh")

        //val modelGroup = ui.createGroup("modelGroup")
        //val modelView = ui.show(modelGroup, model, "model")

        val sampler = UniformMeshSampler3D(model.reference, numberOfPoints = 5000)
        val points: Seq[Point[_3D]] = sampler.sample.map(pointWithProbability => pointWithProbability._1) // we only want the points

        //We work with the point's ID
        val ptIds = points.map(point => model.reference.pointSet.findClosestPoint(point).id)

        //finds for each point of interest the closest point on the target
        def attributeCorrespondences(movingMesh: TriangleMesh[_3D], ptIds: Seq[PointId]): Seq[(PointId, Point[_3D])] = {
          ptIds.map { id: PointId =>
            val pt = movingMesh.pointSet.point(id)
            val closestPointOnMesh2 = targetMesh.pointSet.findClosestPoint(pt).point
            (id, closestPointOnMesh2)
          }
        }

        val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3))

        def fitModel(correspondences: Seq[(PointId, Point[_3D])]): TriangleMesh[_3D] = {
          val regressionData = correspondences.map(correspondence =>
            (correspondence._1, correspondence._2, littleNoise)
          )
          val posterior = model.posterior(regressionData.toIndexedSeq)
          posterior.mean
        }

        //we iterate the procedure
        def nonrigidICP(movingMesh: TriangleMesh[_3D], ptIds: Seq[PointId], numberOfIterations: Int): TriangleMesh[_3D] = {
          if (numberOfIterations == 0) movingMesh
          else {
            val correspondences = attributeCorrespondences(movingMesh, ptIds)
            val transformed = fitModel(correspondences)
            nonrigidICP(transformed, ptIds, numberOfIterations - 1)
          }
        }

        val finalFit = nonrigidICP(model.mean, ptIds, 20)

        MeshIO.writeMesh(finalFit, new java.io.File(s"datasets/challenge-data/challengedata/coresponded-full-femurs/meshes/$i.stl")).get
      }

      val meshFiles = new java.io.File("datasets/challenge-data/challengedata/coresponded-full-femurs/meshes/").listFiles
      val (meshes) = meshFiles.map(meshFile => {
        val mesh = MeshIO.readMesh(meshFile).get
        (mesh)
      })
      val meshes2 : Array[TriangleMesh[_3D]] = meshes

      val reference = MeshIO.readMesh(new java.io.File("data/femur.stl")).get

      val dc = DataCollection.fromTriangleMesh3DSequence(reference, meshes2)
      val modelFromDataCollection = PointDistributionModel.createUsingPCA(dc)

      val modelGroup2 = ui.createGroup("modelGroup2")
      ui.show(modelGroup2, modelFromDataCollection, "PCAModel")
      StatisticalModelIO.writeStatisticalTriangleMeshModel3D(modelFromDataCollection, new java.io.File("datasets/challenge-data/challengedata/GaussianProcessModel/PCAModel.h5"))
    }
}