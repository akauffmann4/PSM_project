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

      val meshFiles = new java.io.File("datasets/challenge-data/challengedata/full-femurs/meshes/").listFiles
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
      var landmarksFiles = new java.io.File("datasets/challenge-data/challengedata/full-femurs/landmarks/").listFiles
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
        val landmarkAligned = landmarks(i).map(lm => lm.transform(rigidTrans))
        val meshView = ui.show(dsGroup2, meshAligned, "meshAligned")

        LandmarkIO.writeLandmarksJson(landmarkAligned, new java.io.File("datasets/challenge-data/challengedata/aligned-full-femurs/landmarks/",landmarksFiles(i).getName)).get
        MeshIO.writeMesh(meshAligned, new java.io.File("datasets/challenge-data/challengedata/aligned-full-femurs/meshes/",meshFiles(i).getName)).get

        i+=1
      }

    }
}