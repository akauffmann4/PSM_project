import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.geometry._
import scalismo.io.{LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel3D}
import scalismo.mesh._
import scalismo.statisticalmodel.{GaussianProcess3D, LowRankGaussianProcess, PointDistributionModel}
import scalismo.ui.api._
//import scalismo.registration._
import scalismo.registration.LandmarkRegistration

object BuildTheModel {

    def main(args: Array[String]) : Unit = {
      implicit val rng = scalismo.utils.Random(42L)
      val ui = ScalismoUI()
      val referenceMesh = MeshIO.readMesh( new java.io.File("data/femur.stl")).get
      val kernelx = GaussianKernel3D(100) * 40// * GaussianKernel3D(50) * 20
      val kernely = GaussianKernel3D(100) * 40
      val kernelz = GaussianKernel3D(100) * 160
      val covarianceFun = DiagonalKernel3D(kernelx, kernely, kernelz)
      val gp = GaussianProcess3D[EuclideanVector[_3D]](covarianceFun)
      val interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
      val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(referenceMesh, gp, relativeTolerance = 1e-2, interpolator)
      val model = PointDistributionModel(referenceMesh, lowRankGP)
      val sample3 = model.sample()
      ui.show(ui.createGroup("model"), model, "gp-model")
      StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, new java.io.File("datasets/challenge-data/challengedata/GaussianProcessModel/GaussianProcessModel.h5"))
    }
}