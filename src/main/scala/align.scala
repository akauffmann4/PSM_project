import scalismo.ui.api._
import scalismo.geometry._
import scalismo.common._
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.mesh._
import scalismo.io.{LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.statisticalmodel._
//import scalismo.registration._
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.dataset._
import scalismo.numerics.PivotedCholesky.RelativeTolerance

object align {

    def main(args: Array[String]) : Unit = {
      scalismo.initialize()
      implicit val rng = scalismo.utils.Random(42)

      val ui = ScalismoUI()

      val dsGroup = ui.createGroup("datasets")

      val meshFiles = new java.io.File("datasets/challenge-data/challengedata/full-femurs/meshes/").listFiles.take(10)
      val (meshes, meshViews) = meshFiles.map(meshFile => {
        val mesh = MeshIO.readMesh(meshFile).get
        val meshView = ui.show(dsGroup, mesh, "mesh")
        (mesh, meshView) // return a tuple of the mesh and the associated view
      }) .unzip // take the tuples apart, to get a sequence of meshes and one of meshViews

      val reference = new java.io.File("data/femur.stl")
      val toAlign : Array[TriangleMesh[_3D]] = meshes

      val refLandmarks: Seq[Landmark[_3D]] =
        LandmarkIO.readLandmarksJson3D(new java.io.File("data/femur.json")).get

      /*val landmarks: Array[Seq[Landmark[_3D]]] =
        LandmarkIO.readLandmarksJson3D(new java.io.File("datasets/challenge-data/challengedata/full-femurs/landmarks/").listFiles).get
      */
      var landmarksFiles = new java.io.File("datasets/challenge-data/challengedata/full-femurs/landmarks/").listFiles.take(10)
      //var refLandmarks = new java.io.File("data/femur.json")

      val landmarks: Array[Seq[Landmark[_3D]]]= landmarksFiles.map { file =>
        LandmarkIO.readLandmarksJson3D(file).get
      }

      var i = 0

      val dsGroup2 = ui.createGroup("datasetsAligned")
      val alignedMeshes = toAlign.map { mesh =>
        //val landmarks = pointIds.map{id => Landmark("L_"+id, mesh.pointSet.point(PointId(id)))}
        val rigidTrans = LandmarkRegistration.rigid3DLandmarkRegistration(landmarks(i), refLandmarks, center = Point(0,0,0))
        val meshAligned = mesh.transform(rigidTrans)
        i+=1
        val meshView = ui.show(dsGroup2, meshAligned, "meshAligned")
      }
      /*val dc = DataCollection.fromTriangleMesh3DSequence(reference, alignedMeshes)
      val modelFromDataCollection = PointDistributionModel.createUsingPCA(dc)

      val modelGroup2 = ui.createGroup("modelGroup2")
      ui.show(modelGroup2, modelFromDataCollection, "ModelDC")

      val dcWithGPAAlignedShapes = DataCollection.gpa(dc)
      val modelFromDataCollectionGPA = PointDistributionModel.createUsingPCA(dcWithGPAAlignedShapes)

      val modelGroup3 = ui.createGroup("modelGroup3")
      ui.show(modelGroup3, modelFromDataCollectionGPA, "ModelDCGPA")*/
    }
}