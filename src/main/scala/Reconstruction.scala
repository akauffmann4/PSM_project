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
    //TO-DO: consider all partial samples
    val PCAModel = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new java.io.File("datasets/challenge-data/challengedata/GaussianProcessModel/PCAModel.lefile")).get
    val targetTest = MeshIO.readMesh(new java.io.File("datasets/challenge-data/challengedata/partial-femurs/51.stl")).get

    val partial = ui.createGroup("modelGroup")
    ui.show(partial, targetTest, "imageTest")

    //val PCAmodel = ui.createGroup("modelGroup")
    //ui.show(PCAmodel, PCAModel, "model")

    //Selects the points for which we want to find the correspondences - uniformly distributed on the surface
    val sampler = UniformMeshSampler3D(PCAModel.reference, numberOfPoints = 5000)
    val points: Seq[Point[_3D]] = sampler.sample.map(pointWithProbability => pointWithProbability._1) // we only want the points

    //Uses point ids of the sampled points
    val ptIds = points.map(point => PCAModel.reference.pointSet.findClosestPoint(point).id)

    //Finds for each point of interest the closest point on the target
    def attributeCorrespondences(movingMesh: TriangleMesh[_3D], ptIds: Seq[PointId]): Seq[(PointId, Point[_3D])] = {
      ptIds.map { id: PointId =>
        val pt = movingMesh.pointSet.point(id)
        val closestPointOnMesh2 = targetTest.pointSet.findClosestPoint(pt).point
        (id, closestPointOnMesh2)
      }
    }

    //Uses the correspondences found to compute a Gaussian process regression
    val correspondences = attributeCorrespondences(PCAModel.mean, ptIds)
    val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3))

    def fitModel(correspondences: Seq[(PointId, Point[_3D])]): TriangleMesh[_3D] = {
      val regressionData = correspondences.map(correspondence =>
        (correspondence._1, correspondence._2, littleNoise)
      )
      val posterior = PCAModel.posterior(regressionData.toIndexedSeq)
      posterior.mean
    }

    val fit = fitModel(correspondences)
    val resultGroup = ui.createGroup("results")
    val fitResultView = ui.show(resultGroup, fit, "fit")

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
  val finalFit = nonrigidICP(PCAModel.mean, ptIds, 30)
  //TO-DO: store all reconstructed samples
  ui.show(resultGroup, finalFit, "final fit")
}

}