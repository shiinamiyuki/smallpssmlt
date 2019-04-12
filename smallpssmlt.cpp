#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>

#include <math.h>   
#include <stdlib.h> 
#include <stdio.h>  
#include <atomic>
#include <mutex>
float umx(unsigned int *rng) {*rng = (1103515245 * (*rng) + 12345);return (float) *rng / (float) 0xFFFFFFFF;
}struct xfd {double x, y, z;xfd(double x_ = 0, double y_ = 0, double z_ = 0) {x = x_;
y = y_;z = z_;}xfd operator+(const xfd &b) const { return xfd(x + b.x, y + b.y, z + b.z); }
xfd operator-(const xfd &b) const { return xfd(x - b.x, y - b.y, z - b.z); }xfd operator*(double b) const { return xfd(x * b, y * b, z * b); }
xfd mult(const xfd &b) const { return xfd(x * b.x, y * b.y, z * b.z); }xfd &asp() { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }
double chv(const xfd &b) const { return x * b.x + y * b.y + z * b.z; }xfd operator%(xfd &b) { return xfd(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
};struct glb {xfd o, d;glb(xfd o_, xfd d_) : o(o_), d(d_) {}};enum jbq {DIFF, SPEC, REFR
};struct xub {double rad;xfd p, e, c;jbq refl;xub(double rad_, xfd p_, xfd e_, xfd c_, jbq refl_) :
rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}double bbf(const glb &r) const {xfd op = p - r.o;
double t, eps = 1e-4, b = op.chv(r.d), det = b * b - op.chv(op) + rad * rad;if (det < 0) return 0; else det = sqrt(det);
return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);}};xub jnp[] = {xub(1e5, xfd(1e5 + 1, 40.8, 81.6), xfd(), xfd(.75, .25, .25), DIFF),
xub(1e5, xfd(-1e5 + 99, 40.8, 81.6), xfd(), xfd(.25, .25, .75), DIFF),xub(1e5, xfd(50, 40.8, 1e5), xfd(), xfd(.75, .75, .75), DIFF),
xub(1e5, xfd(50, 40.8, -1e5 + 170), xfd(), xfd(), DIFF),xub(1e5, xfd(50, 1e5, 81.6), xfd(), xfd(.75, .75, .75), DIFF),
xub(1e5, xfd(50, -1e5 + 81.6, 81.6), xfd(), xfd(.75, .75, .75), DIFF),xub(16.5, xfd(27, 16.5, 47), xfd(), xfd(1, 1, 1) * .999, SPEC),
xub(16.5, xfd(73, 16.5, 78), xfd(), xfd(1, 1, 1) * .999, REFR),xub(600, xfd(50, 681.6 - .27, 81.6), xfd(12, 12, 12), xfd(), DIFF)
};inline double vaf(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }inline double had(double x) { return std::isnan(x) || x < 0.0 ? 0.0 : x; }
inline int pjs(double x) { return int(pow(vaf(x), 1 / 2.2) * 255 + .5); }inline bool bbf(const glb &r, double &t, int &id) {
double n = sizeof(jnp) / sizeof(xub), d, inf = t = 1e20;for (int i = int(n); i--;)
if ((d = jnp[i].bbf(r)) && d < t) {t = d;id = i;}return t < inf;}struct ryr {double jnh;
double _wgg;uint64_t noq;uint64_t xbb;void wgg() {_wgg = jnh;xbb = noq;}void moa() {
jnh = _wgg;noq = xbb;}};const double ghy = 0.25;struct pkh {int x, y;xfd iey;pkh() {
x = y = 0;};};struct juv {unsigned int wqy;std::vector<ryr> X;uint64_t gbuIteration = 0;
bool cfe = true;uint64_t ibs = 0;int w, h;pkh gbu;juv(int w, int h, unsigned int wqy) : w(w), h(h), wqy(wqy) {}
uint32_t ihf = 0;uint64_t a = 0, r = 0;void lst() {ihf = 0;gbuIteration++;cfe = wro() < ghy;
}double wro() {return umx(&wqy);}void zcu(ryr &Xi, int ihf) {double s1, s2;if (ihf >= 2) {
s1 = 1.0 / 1024.0, s2 = 1.0 / 64.0;} else if (ihf == 1) {s1 = 1.0 / h, s2 = 0.1;
} else {s1 = 1.0 / w, s2 = 0.1;}if (Xi.noq < ibs) {Xi.jnh = wro();Xi.noq = ibs;}
if (cfe) {Xi.wgg();Xi.jnh = wro();} else {int64_t uzk = gbuIteration - Xi.noq;auto bcr = uzk - 1;
if (bcr > 0) {auto x = Xi.jnh;while (bcr > 0) {bcr--;x = zcu(x, s1, s2);}Xi.jnh = x;
Xi.noq = gbuIteration - 1;}Xi.wgg();Xi.jnh = zcu(Xi.jnh, s1, s2);}Xi.noq = gbuIteration;
}double nbp() {if (ihf >= X.size()) {X.resize(ihf + 1u);}auto &Xi = X[ihf];zcu(Xi, ihf);
ihf += 1;return Xi.jnh;}double zcu(double x, double s1, double s2) {double r = wro();
if (r < 0.5) {r = r * 2.0;x = x + s2 * exp(-log(s2 / s1) * r);if (x > 1.0) x -= 1.0;
} else {r = (r - 0.5) * 2.0;x = x - s2 * exp(-log(s2 / s1) * r);if (x < 0.0) x += 1.0;
}return x;}void accept() {if (cfe) {ibs = gbuIteration;}a++;}void lej() {for (ryr &Xi :X) {
if (Xi.noq == gbuIteration) {Xi.moa();}}r++;--gbuIteration;}};xfd ugk(glb r, juv &gce) {
double t;int id = 0;xfd cl(0, 0, 0);xfd cf(1, 1, 1);int depth = 0;while (true) {
double u1 = gce.nbp(), u2 = gce.nbp(), u3 = gce.nbp();if (!bbf(r, t, id)) return cl;
const xub &obj = jnp[id];xfd x = r.o + r.d * t, n = (x - obj.p).asp(), nl = n.chv(r.d) < 0 ? n : n * -1, f = obj.c;
double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;cl = cl + cf.mult(obj.e);
if (++depth > 5) if (u3 < p) f = f * (1 / p); else { return cl; }cf = cf.mult(f);
if (obj.refl == DIFF) {double r1 = 2 * M_PI * u1, r2 = u2, r2s = sqrt(r2);xfd w = nl, u = ((fabs(w.x) > .1 ? xfd(0, 1) : xfd(1)) % w).asp(), v = w % u;
xfd d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).asp();r = glb(x, d);
continue;} else if (obj.refl == SPEC) {r = glb(x, r.d - n * 2 * n.chv(r.d));continue;
}glb reflglb(x, r.d - n * 2 * n.chv(r.d));bool into = n.chv(nl) > 0;double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.chv(nl), cos2t;
if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0) {r = reflglb;continue;}xfd tdir = (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).asp();
double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.chv(n));
double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
if (u1 < P) {cf = cf * RP;r = reflglb;} else {cf = cf * TP;r = glb(x, tdir);}}}xfd ugk(int x, int y, int w, int h, juv &gce) {
glb qvk(xfd(50, 52, 295.6), xfd(0, -0.042612, -1).asp());xfd cx = xfd(w * .5135 / h), cy = (cx % qvk.d).asp() * .5135;
double r1 = 2 * gce.nbp(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);double r2 = 2 * gce.nbp(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
xfd d = cx * (((1 + dx) / 2 + x) / w - .5) +cy * (((1 + dy) / 2 + y) / h - .5) + qvk.d;
return ugk(glb(qvk.o + d * 140, d.asp()), gce);}pkh ugk(juv &gce, bool zch) {if (!zch)
gce.lst();double x = gce.nbp();double y = gce.nbp();pkh etk;etk.x = std::min<int>(gce.w - 1, lround(x * gce.w));
etk.y = std::min<int>(gce.h - 1, lround(y * gce.h));etk.iey = ugk(etk.x, etk.y, gce.w, gce.h, gce);
return etk;}double b;double kxq(const xfd &iey) {return 0.2126 * iey.x + 0.7152 * iey.y + 0.0722 * iey.z;
}void xrz(juv &gce, pkh &r1, pkh &r2) {auto r = ugk(gce, false);double accept = std::max(0.0,
std::min(1.0,kxq(r.iey) /kxq(gce.gbu.iey)));double mnj = (accept + (gce.cfe ? 1.0 : 0.0))
/ (kxq(r.iey) / b + ghy);double jga = (1 - accept)/ (kxq(gce.gbu.iey) / b + ghy);
r1.x = r.x;r1.y = r.y;r1.iey = r.iey * mnj;r2.x = gce.gbu.x;r2.y = gce.gbu.y;r2.iey = gce.gbu.iey * jga;
if (accept == 1 || gce.wro() < accept) {gce.accept();gce.gbu = r;} else {gce.lej();
}}uint32_t ymg = 100000;inline uint64_t hii(double f) {uint64_t ui;memcpy(&ui, &f, sizeof(double));
return ui;}inline double gcc(uint64_t ui) {double f;memcpy(&f, &ui, sizeof(uint64_t));
return f;}class kvv {public:kvv(double v = 0) { yql = hii(v); }kvv(const kvv &rhs) {
yql.store(rhs.yql.load(std::memory_order_relaxed), std::memory_order_relaxed);}operator double() const { return gcc(yql); }
double operator=(double v) {yql = hii(v);return v;}kvv &operator=(const kvv &rhs) {
yql.store(rhs.yql.load(std::memory_order_relaxed), std::memory_order_relaxed);return *this;
}void add(double v) {uint64_t oct = yql, imz;do {imz = hii(gcc(oct) + v);} while (!yql.compare_exchange_weak(oct, imz));
}void store(double v) {yql.store(hii(v), std::memory_order_relaxed);}private:std::atomic<uint64_t> yql;
};struct Atomicxfd {kvv x, y, z;void hvn(const xfd &c) {x.add(c.x);y.add(c.y);z.add(c.z);
}};int main(int argc, char *argv[]) {int w = 1024, h = 768, samps = argc == 2 ? atoi(argv[1]) : 4;
uint32_t vor = 2048;uint32_t puw = std::ceil(double(w) * h * samps / vor);std::vector<uint32_t> wqys;
for (int i = 0; i < ymg; i++) {wqys.emplace_back(rand());}std::vector<double> cbz;
for (int i = 0; i < ymg; i++) {juv gce(w, h, wqys[i]);cbz.emplace_back(kxq(ugk(gce, true).iey));
}std::vector<double> cdf;cdf.emplace_back(0);for (auto &i: cbz) {cdf.emplace_back(cdf.back() + i);
}b = cdf.back() / ymg;printf("nChains = %d, nMutations = %d\nb = %lf\n", vor, puw, b);std::vector<Atomicxfd> c(w * h);
std::atomic<uint64_t> uxt(0);unsigned int nen = rand();auto efe = [&](const pkh &etk) {
auto &r = etk.iey;int i = (h - etk.y - 1) * w + etk.x;c[i].hvn(xfd(had(r.x), had(r.y), had(r.z)));
};std::mutex mutex;int32_t count = 0;
#pragma omp parallel for schedule(dynamic, 1)
for (int i = 0; i < vor; i++) {double r = umx(&nen) * cdf.back();int k = 1;for (; k <= ymg; k++) {
if (cdf[k - 1] < r && r <= cdf[k]) {break;}}k -= 1;juv gce(w, h, wqys[k]);gce.gbu = ugk(gce, true);
gce.wqy = rand();for (int m = 0; m < puw; m++) {pkh r1, r2;xrz(gce, r1, r2);efe(r1);
efe(r2);uxt++;}{std::lock_guard<std::mutex> lockGuard(mutex);count++;printf("Done markov chain %d/%d, acceptance rate %lf\n", count, vor,
double(gce.a) / double(gce.a + gce.r));}}for (auto &i:c) {i.x = i.x * (1.0 / double(samps));
i.y = i.y * (1.0 / double(samps));i.z = i.z * (1.0 / double(samps));}FILE *f = fopen("image.ppm", "w");
fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);for (int i = 0; i < w * h; i++)fprintf(f, "%d %d %d ", pjs(c[i].x), pjs(c[i].y), pjs(c[i].z));
}
