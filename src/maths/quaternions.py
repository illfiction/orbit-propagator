import numpy as np


class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

    def as_array(self):
        return np.array([self.w, self.x, self.y, self.z])

    # Norm
    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    # Normalize
    def normalize(self):
        n = self.norm()
        if n > 0:
            self.w /= n
            self.x /= n
            self.y /= n
            self.z /= n
        return self

    # Conjugate
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    # Inverse
    def inverse(self):
        n2 = self.norm() ** 2
        return self.conjugate() * (1.0 / n2)

    # Scalar multiplication
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            # Hamilton product
            w1, x1, y1, z1 = self.as_array()
            w2, x2, y2, z2 = other.as_array()
            return Quaternion(
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        else:
            # Scalar multiply
            return Quaternion(
                self.w * other, self.x * other, self.y * other, self.z * other
            )

    # Rotate vector
    def rotate_vector(self, v):
        qv = Quaternion(0, *v)
        return (self * qv * self.inverse()).vector_part()

    def vector_part(self):
        return np.array([self.x, self.y, self.z])

    # From angular velocity
    @staticmethod
    def from_omega(omega, dt):
        # omega is angular velocity vector in rad/s
        theta = np.linalg.norm(omega) * dt
        if theta == 0:
            return Quaternion()
        axis = omega / np.linalg.norm(omega)
        half_theta = theta / 2
        w = np.cos(half_theta)
        xyz = axis * np.sin(half_theta)
        return Quaternion(w, *xyz).normalize()
