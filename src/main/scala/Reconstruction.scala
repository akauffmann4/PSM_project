import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.{DiscreteField3D, PointId, UnstructuredPointsDomain3D}
import scalismo.geometry._
import scalismo.io.{ActiveShapeModelIO, ImageIO, LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.mesh._
import scalismo.numerics.UniformMeshSampler3D
import scalismo.ui.api._
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.MultivariateNormalDistribution
import scalismo.statisticalmodel.asm.{FittingConfiguration, ModelTransformations, NormalDirectionSearchPointSampler}
import scalismo.transformations.{Rotation3D, Translation3D, TranslationAfterRotation3D}
import scalismo.utils.Random.implicits.randomGenerator

object Reconstruction {

  def main(args: Array[String]): Unit = {

    val ui = ScalismoUI()
    for (i <- 47 until 57
         ) {
      //TO-DO: consider all partial samples
      val PCAModel = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new java.io.File("datasets/challenge-data/challengedata/GaussianProcessModel/PCAModel.h5")).get
      val targetTest = MeshIO.readMesh(new java.io.File(s"datasets/challenge-data/challengedata/partial-femurs/$i.stl")).get

      val partial = ui.createGroup("modelGroup")
      ui.show(partial, targetTest, "imageTest")

      val PCAmodel = ui.createGroup("modelGroup")
      ui.show(PCAmodel, PCAModel, "model")
      //var lastmesh = PCAModel.mean
      //Selects the points for which we want to find the correspondences - uniformly distributed on the surface
      val sampler = UniformMeshSampler3D(targetTest, numberOfPoints = 50000)
      val points: Seq[Point[_3D]] = sampler.sample.map(pointWithProbability => pointWithProbability._1)//.filter(pt => pt.z<= 2.0) // we only want the points

      //Uses point ids of the sampled points
      val ptIds = points.map(point => targetTest.pointSet.findClosestPoint(point).id)

      //Finds for each point of interest the closest point on the target
      val Vectors = ui.createGroup("Vectors")
      def attributeCorrespondences(movingMesh: TriangleMesh[_3D], ptIds: Seq[PointId]): Seq[(PointId, Point[_3D])] = {
        ptIds.map { id: PointId =>
          val pt = targetTest.pointSet.point(id)
          val closestPointOnMesh2 = movingMesh.pointSet.findClosestPoint(pt).id

          (closestPointOnMesh2, pt)
        }
      }

      val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3)*5.0)

      def fitModel(correspondences: Seq[(PointId, Point[_3D])]): TriangleMesh[_3D] = {
        val regressionData = correspondences.map(correspondence =>
          (correspondence._1, correspondence._2, littleNoise)
        )
        val posterior = PCAModel.posterior(regressionData.toIndexedSeq)
        posterior.mean
      }


      //Iterates the procedure
      def nonrigidICP(movingMesh: TriangleMesh[_3D], ptIds: Seq[PointId], numberOfIterations: Int): TriangleMesh[_3D] = {
        if (numberOfIterations == 0) movingMesh
        else {
          val correspondences = attributeCorrespondences(movingMesh, ptIds )
          val transformed = fitModel(correspondences)
          //lastmesh = transformed
          //ui.show(resultGroup, transformed, numberOfIterations.toString)
          nonrigidICP(transformed, ptIds, numberOfIterations - 1)
        }
      }

      //Repeats the fitting steps iteratively for 30 times
      val finalFit = nonrigidICP(PCAModel.mean, ptIds, 60)
      //TO-DO: store all reconstructed samples
      val resultGroup = ui.createGroup("results")
      //ui.show(resultGroup, finalFit, "final fit")
      MeshIO.writeMesh(finalFit, new java.io.File(s"datasets/challenge-data/challengedata/reconstructed-femurs/$i.stl")).get
    }
}

}