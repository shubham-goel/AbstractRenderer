#include <iostream>
#include <fstream>
#include <cassert>
#include <bits/stdc++.h>
#include "CImg.h"
#include "boost/program_options.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_USE_DOUBLE
#include "tiny_obj_loader.h"

#define PI 3.14159265359

using namespace std;

class Point3d {
public:
    double x, y, z;
    Point3d():x(0),y(0),z(0){};
    Point3d(double _x, double _y, double _z):x(_x),y(_y),z(_z){};
    void print(ostream &s, string end = "") const {
        s << "(" << x << "," << y << "," << z << ")" << end;
    }
    void operator+=(const Point3d& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
    }
    void operator-=(const Point3d& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
    }
    void operator+=(double rhs) {
        x += rhs;
        y += rhs;
        z += rhs;
    }
    void operator-=(double rhs) {
        x -= rhs;
        y -= rhs;
        z -= rhs;
    }
    void operator*=(double rhs) {
        x *= rhs;
        y *= rhs;
        z *= rhs;
    }
    Point3d operator+(const Point3d& rhs) const {
        double _x = x+rhs.x;
        double _y = y+rhs.y;
        double _z = z+rhs.z;
        return Point3d(_x,_y,_z);
    }
    Point3d operator-(const Point3d& rhs) const {
        double _x = x-rhs.x;
        double _y = y-rhs.y;
        double _z = z-rhs.z;
        return Point3d(_x,_y,_z);
    }
    Point3d operator+(double rhs) const {
        double _x = x+rhs;
        double _y = y+rhs;
        double _z = z+rhs;
        return Point3d(_x,_y,_z);
    }
    Point3d operator-(double rhs) const {
        double _x = x-rhs;
        double _y = y-rhs;
        double _z = z-rhs;
        return Point3d(_x,_y,_z);
    }
    Point3d operator-() const {
        double _x = -x;
        double _y = -y;
        double _z = -z;
        return Point3d(_x,_y,_z);
    }
    Point3d operator*(double rhs) const {
        double _x = x*rhs;
        double _y = y*rhs;
        double _z = z*rhs;
        return Point3d(_x,_y,_z);
    }
    Point3d operator/(double rhs) const {
        double _x = x/rhs;
        double _y = y/rhs;
        double _z = z/rhs;
        return Point3d(_x,_y,_z);
    }
    Point3d cross(const Point3d& rhs) const {
        double _x = y*rhs.z - z*rhs.y;
        double _y = z*rhs.x - x*rhs.z;
        double _z = x*rhs.y - y*rhs.x;
        return Point3d(_x,_y,_z);
    }
    double dot(const Point3d& rhs) const {
        double _x = x * rhs.x;
        double _y = y * rhs.y;
        double _z = z * rhs.z;
        return _x + _y + _z;
    }
    double norm() const {
        return sqrt(x*x + y*y + z*z);
    }
};

class Face {
public:
    int v0, v1, v2;
    Face():v0(-1),v1(-1),v2(-1){};
    Face(int _v0, int _v1, int _v2):v0(_v0),v1(_v1),v2(_v2){};
    void print(ostream &s, string end = "") {
        s << "(" << v0 << "," << v1 << "," << v2 << ")" << end;
    }
};

class Matrix33 {
public:
    Point3d r0;
    Point3d r1;
    Point3d r2;
    Matrix33() {};
    Matrix33(Point3d _r0, Point3d _r1, Point3d _r2):
        r0(_r0),r1(_r1),r2(_r2) {};
    Point3d col0() const {
        return Point3d(r0.x, r1.x, r2.x);
    }
    Point3d col1() const {
        return Point3d(r0.y, r1.y, r2.y);
    }
    Point3d col2() const {
        return Point3d(r0.z, r1.z, r2.z);
    }
    Matrix33 matmul(const Matrix33& rhs) const {
        Point3d _r0(r0.dot(rhs.col0()), r0.dot(rhs.col1()), r0.dot(rhs.col2()));
        Point3d _r1(r1.dot(rhs.col0()), r1.dot(rhs.col1()), r1.dot(rhs.col2()));
        Point3d _r2(r2.dot(rhs.col0()), r2.dot(rhs.col1()), r2.dot(rhs.col2()));
        return Matrix33(_r0, _r1, _r2);
    }
    Point3d dot(const Point3d& rhs) const {
        return Point3d(r0.dot(rhs), r1.dot(rhs), r2.dot(rhs));
    }
    Matrix33 t() { // Transpose
        Point3d _rx(r0.x, r1.x, r2.x);
        Point3d _ry(r0.y, r1.y, r2.y);
        Point3d _rz(r0.z, r1.z, r2.z);
        return Matrix33(_rx, _ry, _rz);
    }
    Matrix33 inverse() {
        double s00 = r1.y*r2.z - r2.y*r1.z;
        double s01 = r0.z*r2.y - r2.z*r0.y;
        double s02 = r0.y*r1.z - r1.y*r0.z;

        double s10 = r1.z*r2.x - r2.z*r1.x;
        double s11 = r0.x*r2.z - r2.x*r0.z;
        double s12 = r0.z*r1.x - r1.z*r0.x;

        double s20 = r1.x*r2.y - r2.x*r1.y;
        double s21 = r0.y*r2.x - r2.y*r0.x;
        double s22 = r0.x*r1.y - r1.x*r0.y;

        double d = r0.x*s00 + r0.y*s01 + r0.z*s02;

        auto _r0 = Point3d(s00, s01, s02)/d;
        auto _r1 = Point3d(s10, s11, s12)/d;
        auto _r2 = Point3d(s20, s21, s22)/d;

        return Matrix33(_r0, _r1, _r2);
    }
};


struct {
    string mesh;
    double rescale;
    double theta_max;
    double delta_t_max;
    double t_z;
    int img_h;
    int img_w;
    int img_upsample;
    double light_ambient;
    double light_color;
    Point3d light_src;
} CONFIG;

void imshow(vector<vector<double>>& img) {
    int img_h = img.size();
    int img_w = img[0].size();
    cimg_library::CImg<unsigned char> image(img_h,img_w,1,1,0);
    cimg_forXY(image,x,y) { image(x,y) = (int)(min(max(img[y][x],0.),1.)*255); }
    image.display();
}

vector<vector<double>> downsample(vector<vector<double>> image, int factor) {
    int img_h = image.size();
    int img_w = image[0].size();
    assert(img_h%factor==0);
    assert(img_w%factor==0);
    vector<vector<double>> result(img_h/factor, vector<double>(img_w/factor,0));
    for (int i = 0; i < img_h/factor; i++) {
        for (int j = 0; j < img_w/factor; j++) {
            double sum = 0;
            double coeffs = 0;
            for (int ii = 0; ii < factor; ii++) {
                for (int jj = 0; jj < factor; jj++) {
                    double coeff = (ii-factor/2.)*(ii-factor/2.) + (jj-factor/2.)*(jj-factor/2.);
                    coeff = exp(-coeff/(2 * factor/6. * factor/6.));
                    coeffs += coeff;
                    sum += coeff*image[factor*i+ii][factor*j+jj];
                }
            }
            result[i][j] = sum/(coeffs);
        }
    }
    return result;
}

void parse_arguments(int argc, char const *argv[]) {
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Help screen")
      ("mesh", po::value<string>()->required(), "Mesh obj file")
      ("rescale", po::value<float>()->default_value(0), "Mesh rescaling size")
      ("theta_max", po::value<float>()->default_value(90), "Rotation perturbation (in degrees)")
      ("delta_t_max", po::value<float>()->default_value(0.5), "Translation perturbation")
      ("t_z", po::value<float>()->default_value(3), "Mean z translation")
      ("img_h", po::value<int>()->default_value(64), "Image height")
      ("img_w", po::value<int>()->default_value(64), "Image width")
      ("img_upsample", po::value<int>()->default_value(8), "Image upsampling factor")
      ("light_src_x", po::value<float>()->default_value(-1), "Light source intensity")
      ("light_src_y", po::value<float>()->default_value(-1), "Light source intensity")
      ("light_src_z", po::value<float>()->default_value(2.5), "Light source intensity")
      ("light_color", po::value<float>()->default_value(1), "Light source intensity")
      ("light_ambient", po::value<float>()->default_value(0.2), "Ambient Light")
      ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        exit(0);
    }

    CONFIG.mesh             = vm["mesh"].as<string>();
    CONFIG.rescale          = vm["rescale"].as<float>();
    CONFIG.theta_max        = vm["theta_max"].as<float>();
    CONFIG.delta_t_max      = vm["delta_t_max"].as<float>();
    CONFIG.t_z              = vm["t_z"].as<float>();
    CONFIG.img_h            = vm["img_h"].as<int>();
    CONFIG.img_w            = vm["img_w"].as<int>();
    CONFIG.img_upsample     = vm["img_upsample"].as<int>();
    CONFIG.light_ambient    = vm["light_ambient"].as<float>();
    CONFIG.light_color      = vm["light_color"].as<float>();
    double light_src_x      = vm["light_src_x"].as<float>();
    double light_src_y      = vm["light_src_y"].as<float>();
    double light_src_z      = vm["light_src_z"].as<float>();
    CONFIG.light_src        = Point3d(light_src_x, light_src_y, light_src_z);

    cout << "CONFIG.mesh             : " << CONFIG.mesh             << endl;
    cout << "CONFIG.rescale          : " << CONFIG.rescale          << endl;
    cout << "CONFIG.theta_max        : " << CONFIG.theta_max        << endl;
    cout << "CONFIG.delta_t_max      : " << CONFIG.delta_t_max      << endl;
    cout << "CONFIG.t_z              : " << CONFIG.t_z              << endl;
    cout << "CONFIG.img_h            : " << CONFIG.img_h            << endl;
    cout << "CONFIG.img_w            : " << CONFIG.img_w            << endl;
    cout << "CONFIG.img_upsample     : " << CONFIG.img_upsample     << endl;
    cout << "CONFIG.light_ambient    : " << CONFIG.light_ambient    << endl;
    cout << "CONFIG.light_color      : " << CONFIG.light_color      << endl;
    cout << "CONFIG.light_src        : "; CONFIG.light_src.print(cout, "\n");
}

class Mesh {
public:
    vector<Point3d> vertices;
    vector<Face> faces;
    vector<Point3d> normals;
    Mesh(){};
    Mesh(vector<Point3d> _v, vector<Face> _f):
        vertices(_v), faces(_f) {
            // Compute Normals
            for (auto f:faces) {
                auto edge_01 = vertices[f.v1]-vertices[f.v0];
                auto edge_02 = vertices[f.v2]-vertices[f.v0];
                auto normal = edge_01.cross(edge_02);
                normals.push_back(normal/normal.norm());
            }
        };
    Mesh(vector<Point3d> _v, vector<Face> _f, vector<Point3d> _n):
        vertices(_v), faces(_f), normals(_n) {};
    void print(ostream& s) const {
        s << "Vertices:\n";
        for(auto v:vertices) {
            v.print(s, "\n");
        }
        s << "\n\n";

        s << "Faces:\n";
        for(auto f:faces) {
            f.print(s, "\n");
        }
        s << "\n\n";

        s << "Normals:\n";
        for(auto v:normals) {
            v.print(s, "\n");
        }
        s << "\n\n";
    }
    void rotate(Matrix33 R) {
        for(auto&v : vertices) {
            v = R.dot(v);
        }
        for(auto&n : normals) {
            n = R.dot(n);
        }
    }
    void translate(Point3d t) {
        for(auto&v : vertices) {
            v += t;
        }
    }
    void rescale(double scale) {
        if(scale==0)
            return;
        double max_v = 0;
        for(auto v:vertices) {
            max_v = max(max_v, abs(v.x));
            max_v = max(max_v, abs(v.y));
            max_v = max(max_v, abs(v.z));
        }
        cout << "Rescaling " << max_v << " -> " << scale << endl;
        for(auto&v:vertices) {
            v *= scale/max_v;
        }
    }
    vector<vector<double>> render(int img_h, int img_w,
                            double light_ambient = 0.2,
                            double light_color = 1,
                            Point3d light_src = Point3d(-1,-1,2),
                            int img_upsample = 4) {

        cout << "Rendering Image(" << img_h << "," << img_w << ") upsampled by "
            << img_upsample << "... " << endl;

        img_h *= img_upsample;
        img_w *= img_upsample;

        vector<vector<double>> image(img_h, vector<double>(img_w, 0));

        double img_wf = (float)img_w;
        double img_hf = (float)img_h;
        Matrix33 K(Point3d(img_wf,  0,      img_wf/2),
                    Point3d(0,      img_hf, img_hf/2),
                    Point3d(0,      0,      1));
        auto K_inv = K.inverse();

        vector<Matrix33> v012;
        vector<Matrix33> v012_inv;
        for(auto&f:faces) {
            Matrix33 mm(vertices[f.v0], vertices[f.v1], vertices[f.v2]);
            v012.push_back(mm.t());
        }
        cout << "Precomputing inverses... ";
        for (int i = 0, done=-1; i < v012.size(); i++) {
            int doing =  (int)(i/((float)v012.size())*10);
            if(doing > done) {
                done = doing;
                cout << done*10 << " ";
                cout.flush();
            }
            v012_inv.push_back(v012[i].inverse());
        }
        cout << endl;

        cout << "Coloring... ";
        int done = -1;
        for (int i = 0, done=-1; i < img_h; i++) {
            int doing =  (int)(i/img_hf*10);
            if(doing > done) {
                done = doing;
                cout << done*10 << " ";
                cout.flush();
            }
            for (int j = 0; j < img_w; j++) {
                Point3d ray(j,i,1);
                // Find point of interesection with each face
                //      ray     = K @ (v0;v1;v2) @ (a,b,c)
                // =>   (a,b,c) = (v0;v1;v2)_inv @ K_inv @ ray
                vector<Point3d> coeffs;
                for(auto&mm_inv:v012_inv) {
                    auto abc = mm_inv.dot(K_inv.dot(ray));
                    coeffs.push_back(abc);
                }


                // //////////////////////////////////////////////////////
                // /////////////// START: Needs Abstraction /////////////
                // //////////////////////////////////////////////////////

                // Distances and colors of all intersecting triangles
                vector<double> distances;
                vector<double> colors;

                for (int f = 0; f < faces.size(); f++) {
                    auto abc = coeffs[f];
                    double sum = abc.x + abc.y + abc.z;
                    if(sum != 0) {
                        abc = abc/sum;
                        if(abc.x>=0 && abc.y>=0 && abc.z>=0) {
                            // Ray intersects triangle
                            auto P = v012[f].dot(abc);
                            double dist = P.x*P.x + P.y*P.y + P.z*P.z;
                            distances.push_back(dist);

                            auto light_ray = light_src-P;
                            light_ray = light_ray/light_ray.norm();
                            double ii = normals[f].dot(light_ray);
                            // ii = max(ii, 0.);
                            ii = (ii+1)/2.;
                            double color = ii*light_color + (1-ii)*light_ambient;
                            colors.push_back(color);
                        }
                    }
                }

                // Choose color corresponding to minimum distance
                double min_d = 1000;    // @TODO: Fix
                double min_c = 0;
                for (int fi = 0; fi < distances.size(); fi++) {
                    if(distances[fi] < min_d) {
                        min_c = colors[fi];
                        min_d = distances[fi];
                    }
                }
                // ////////////////////////////////////////////////////
                // /////////////// END: Needs Abstraction /////////////
                // ////////////////////////////////////////////////////

                image[i][j] = min_c;
            }
        }
        cout << endl;


        image = downsample(image, img_upsample);

        return image;
    }
};

Matrix33 R_from_thetaX(double theta) {
    double c = cos(theta);
    double s = sin(theta);
    auto r0 = Point3d(1,0,0);
    auto r1 = Point3d(0,c,-s);
    auto r2 = Point3d(0,s,c);
    return Matrix33(r0,r1,r2);
}
Matrix33 R_from_thetaY(double theta) {
    double c = cos(theta);
    double s = sin(theta);
    auto r0 = Point3d(c,0,s);
    auto r1 = Point3d(0,1,0);
    auto r2 = Point3d(-s,0,c);
    return Matrix33(r0,r1,r2);
}
Matrix33 R_from_thetaZ(double theta) {
    double c = cos(theta);
    double s = sin(theta);
    auto r0 = Point3d(c,-s,0);
    auto r1 = Point3d(s,c,0);
    auto r2 = Point3d(0,0,1);
    return Matrix33(r0,r1,r2);
}
Matrix33 R_from_thetas(Point3d thetas) {
    auto Rx = R_from_thetaX(thetas.x);
    auto Ry = R_from_thetaY(thetas.y);
    auto Rz = R_from_thetaZ(thetas.z);
    return Rz.matmul(Ry.matmul(Rx));
}

Mesh load_mesh(string inputfile) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str(), NULL, true);

    if (!warn.empty()) {
    std::cout << warn << std::endl;
    }

    if (!err.empty()) {
    std::cerr << err << std::endl;
    }

    if (!ret) {
    exit(1);
    }

    auto shape = shapes[0];
    vector<Point3d> vertices;
    vector<Face> faces;

    assert(attrib.vertices.size()%3==0);
    int N = attrib.vertices.size()/3;
    int M = shape.mesh.num_face_vertices.size();

    // Loop over vertices
    for (size_t i = 0; i < N; i++) {
        double x = attrib.vertices[3*i+0];
        double y = attrib.vertices[3*i+1];
        double z = attrib.vertices[3*i+2];
        Point3d vertex = Point3d(x,y,z);
        vertices.push_back(vertex);
    }

    // Loop over faces
    for (size_t f = 0; f < M; f++) {
        assert(shape.mesh.num_face_vertices[f] == 3);
        int v0 = shape.mesh.indices[3*f + 0].vertex_index;
        int v1 = shape.mesh.indices[3*f + 1].vertex_index;
        int v2 = shape.mesh.indices[3*f + 2].vertex_index;
        Face face = Face(v0,v1,v2);
        faces.push_back(face);
    }

    return Mesh(vertices, faces);
}

int main(int argc, char const *argv[])
{
    parse_arguments(argc, argv);
    // string mesh_file = "3dmodels/box_stack.obj";
    // string mesh_file = "../3dmodels/cuboid.obj";
    // string mesh_file = "../3dmodels/tetrahedron.obj";
    // string mesh_file = "../3dmodels/guitar.obj";

    auto mesh = load_mesh(CONFIG.mesh);
    mesh.rescale(CONFIG.rescale);
    cout << "Num Vertices: " << mesh.vertices.size() << endl;
    cout << "Num Faces:    " << mesh.faces.size() << endl;
    // mesh.print(cout);

    default_random_engine generator;
    uniform_real_distribution<double> distribution(-1.0,1.0);

    while(true) {
        auto mesh_copy = mesh;
        // R, t
        Point3d thetas(distribution(generator)*(CONFIG.theta_max)/180*PI,
                        distribution(generator)*(CONFIG.theta_max)/180*PI,
                        distribution(generator)*(CONFIG.theta_max)/180*PI);
        Point3d t(0,0,CONFIG.t_z);
        Point3d delta_t(distribution(generator)*CONFIG.delta_t_max,
                        distribution(generator)*CONFIG.delta_t_max,
                        distribution(generator)*CONFIG.delta_t_max);
        t += delta_t;

        cout << "thetas: "; thetas.print(cout, "\n");
        cout << "t:      "; t.print(cout, "\n");

        Matrix33 R = R_from_thetas(thetas);

        // Apply R,t
        mesh_copy.rotate(R);
        mesh_copy.translate(t);

        // Render
        auto img = mesh_copy.render(CONFIG.img_h, CONFIG.img_w,
                                    CONFIG.light_ambient,
                                    CONFIG.light_color,
                                    CONFIG.light_src,
                                    CONFIG.img_upsample);
        imshow(img);
    }

    return 0;
}
