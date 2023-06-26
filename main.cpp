#include "kernel.cuh"
#include <iostream>

class Vec3f {
public:
    float x, y, z;

    Vec3f() {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }

    Vec3f(float x, float y, float z) {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    Vec3f(const Vec3f& vec3f) {
        x = vec3f.x;
        y = vec3f.y;
        z = vec3f.z;
    }

    float length() {
        return sqrt((x * x) + (y * y) + (z * z));
    }

    void normalize() {
        float len = length();
        x /= len;
        y /= len;
        z /= len;
    }

    void print() {
        std::cout << "[" << x << ", " << y << ", " << z << "]\n";
    }

    Vec3f operator + (const float val) const {
        return Vec3f(x + val, y + val, z + val);
    }

    Vec3f operator - (const float val) const {
        return Vec3f(x - val, y - val, z - val);
    }

    Vec3f operator * (const float val) const {
        return Vec3f(x * val, y * val, z * val);
    }

    Vec3f operator / (const float val) const {
        return Vec3f(x / val, y / val, z / val);
    }
    
    Vec3f operator + (const Vec3f& vec3f) const {
        return Vec3f(x + vec3f.x, y + vec3f.y, z + vec3f.z);
    }

    Vec3f operator - (const Vec3f& vec3f) const {
        return Vec3f(x - vec3f.x, y - vec3f.y, z - vec3f.z);
    }

    void operator += (const float val) {
        x += val;
        y += val;
        z += val;
    }

    void operator -= (const float val) {
        x -= val;
        y -= val;
        z -= val;
    }

    void operator *= (const float val) {
        x *= val;
        y *= val;
        z *= val;
    }

    void operator /= (const float val) {
        x /= val;
        y /= val;
        z /= val;
    }

    void operator += (const Vec3f vec3f) {
		x += vec3f.x;
		y += vec3f.y;
        z += vec3f.z;
	}

    void operator -= (const Vec3f vec3f) {
        x -= vec3f.x;
        y -= vec3f.y;
        z -= vec3f.z;
    }

    bool operator > (Vec3f vec3f) {
        float len1 = length();
        float len2 = vec3f.length();
        return len1 > len2;
    }

    bool operator < (Vec3f vec3f) {
        float len1 = length();
        float len2 = vec3f.length();
        return len1 < len2;
    }
};

int main() {
    std::cout << sizeof(float) << "\n";
    Vec3f a;
    Vec3f& b = a;
    std::cout << sizeof(b) << "\n";

    return 0;
}