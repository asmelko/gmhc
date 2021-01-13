#include "validator.hpp"

#include <algorithm>
#include <cfloat>
#include <iostream>
#include <map>

#include "../tests/serial_impl.hpp"
#include "kernels.cuh"

using namespace clustering;

void validator::create_clusters(const asgn_t* apriori_assignments)
{
    std::map<asgn_t, std::vector<cluster>> clusters;

    for (csize_t i = 0; i < point_count_; i++)
    {
        auto idx = apriori_assignments[i];

        auto it = clusters.find(idx);

        if (it == clusters.end())
            clusters.emplace(idx, std::vector<cluster>());

        it = clusters.find(idx);

        cluster c;
        c.id = (csize_t)i;
        c.size = (csize_t)1;
        c.mf = (csize_t)1;

        for (csize_t j = 0; j < point_dim_; j++)
            c.centroid.push_back(points_[i * point_dim_ + j]);

        c.icov = unit_matrix_;

        it->second.emplace_back(std::move(c));
    }

    for (auto& cl : clusters)
    {
        clusters_.insert(
            clusters_.end(), std::make_move_iterator(cl.second.begin()), std::make_move_iterator(cl.second.end()));
        apr_sizes_.push_back((csize_t)cl.second.size());
    }
}

void validator::initialize(const float* data_points,
    csize_t data_points_size,
    csize_t data_point_dim,
    csize_t maha_threshold,
    subthreshold_handling_kind subthreshold_kind,
    const asgn_t* apriori_assignments)
{
    points_ = data_points;
    point_count_ = data_points_size;
    point_dim_ = data_point_dim;
    maha_threshold_ = maha_threshold;
    subthreshold_kind_ = subthreshold_kind;
    error_ = false;

    id_ = (asgn_t)point_count_;
    cluster_count_ = point_count_;
    maha_cluster_count_ = 0;

    unit_matrix_ = std::vector<float>(point_dim_ * point_dim_, 0.f);
    for (csize_t j = 0; j < point_dim_; j++)
        unit_matrix_[j * (point_dim_ + 1)] = 1.f;

    asgns_.resize(point_count_);

    if (apriori_assignments)
        create_clusters(apriori_assignments);
    else
        for (csize_t i = 0; i < point_count_; i++)
        {
            cluster c;
            c.id = (asgn_t)i;
            c.size = (csize_t)1;
            c.mf = (csize_t)1;

            for (csize_t j = 0; j < point_dim_; j++)
                c.centroid.push_back(data_points[i * point_dim_ + j]);

            c.icov = unit_matrix_;

            clusters_.emplace_back(std::move(c));
        }

    for (csize_t i = 0; i < point_count_; i++)
        point_asgns_.push_back((asgn_t)i);

    apr_idx_ = 0;
    iteration_ = 1;
}

float eucl_dist(const float* lhs, const float* rhs, csize_t size)
{
    float tmp_dist = 0;
    for (csize_t k = 0; k < size; k++)
    {
        auto tmp = (lhs[k] - rhs[k]);
        tmp_dist += tmp * tmp;
    }
    return std::sqrt(tmp_dist);
}

std::vector<float> mat_vec(const float* mat, float mf, const float* vec, csize_t size)
{
    std::vector<float> res;
    for (csize_t i = 0; i < size; i++)
    {
        res.push_back(0);
        for (csize_t j = 0; j < size; j++)
            res.back() += (mat[i * size + j] / mf) * vec[j];
    }
    return res;
}

float dot(const float* lhs, const float* rhs, csize_t size)
{
    float res = 0;
    for (csize_t i = 0; i < size; i++)
        res += lhs[i] * rhs[i];
    return res;
}

std::vector<float> minus(const float* lhs, const float* rhs, csize_t size)
{
    std::vector<float> res;
    for (csize_t i = 0; i < size; i++)
        res.emplace_back(lhs[i] - rhs[i]);
    return res;
}

float compute_distance(const float* lhs_v,
    const float* lhs_m,
    float lhs_mf,
    const float* rhs_v,
    const float* rhs_m,
    float rhs_mf,
    csize_t size)
{
    float dist = 0;

    {
        auto diff = minus(lhs_v, rhs_v, size);
        auto tmp = mat_vec(lhs_m, lhs_mf, diff.data(), size);
        auto tmp_dist = std::sqrt(dot(tmp.data(), diff.data(), size));
        dist += isnan(tmp_dist) ? eucl_dist(lhs_v, rhs_v, size) : tmp_dist;
    }

    {
        auto diff = minus(lhs_v, rhs_v, size);
        auto tmp = mat_vec(rhs_m, rhs_mf, diff.data(), size);
        auto tmp_dist = std::sqrt(dot(tmp.data(), diff.data(), size));
        dist += isnan(tmp_dist) ? eucl_dist(lhs_v, rhs_v, size) : tmp_dist;
    }

    if (isinf(dist) || isnan(dist))
        return FLT_MAX;
    return dist / 2;
}

csize_t update_asgns(asgn_t* asgns, csize_t count, pasgn_t old_pair, asgn_t id)
{
    csize_t tmp_count = 0;
    for (csize_t i = 0; i < count; i++)
    {
        if (asgns[i] == old_pair.first || asgns[i] == old_pair.second)
        {
            asgns[i] = id;
            tmp_count++;
        }
    }
    return tmp_count;
}

std::vector<float> compute_centroid(
    const float* points, csize_t dim, csize_t size, const asgn_t* assignments, asgn_t cid)
{
    csize_t count = 0;
    std::vector<float> tmp_sum;
    tmp_sum.resize(dim);

    for (csize_t i = 0; i < dim; i++)
        tmp_sum[i] = 0;

    for (csize_t i = 0; i < size; i++)
    {
        if (assignments[i] == cid)
        {
            for (csize_t j = 0; j < dim; j++)
                tmp_sum[j] += points[i * dim + j];
            count++;
        }
    }

    for (csize_t i = 0; i < dim; i++)
        tmp_sum[i] /= count;

    return tmp_sum;
}

void print_pairs(csize_t iteration, const pasgn_t& lhs, const pasgn_t& rhs)
{
    std::cerr << "Iteration " << iteration << ": pairs do not match: " << lhs.first << ", " << lhs.second
              << " =/= " << rhs.first << ", " << rhs.second << std::endl;
}

template<typename T>
void print_arrays(csize_t iteration, const std::string& msg, csize_t size, const T* lhs, const T* rhs)
{
    std::cerr << "Iteration " << iteration << ": " << msg << std::endl;

    for (csize_t j = 0; j < size; j++)
        std::cerr << lhs[j] << " " << rhs[j] << std::endl;
}

bool validator::verify(pasgn_t pair_v, float dist_v, const float* centroid_v, recompute_f recompute)
{
    printf("\r%d ", iteration_);
    fflush(stderr);

    auto [min_pair, new_clust, min_dist] = iterate(std::make_pair(pair_v, dist_v), recompute);

    if (min_pair != pair_v)
    {
        print_pairs(iteration_, min_pair, pair_v);

        error_ = true;
        return false;
    }

    if (float_diff(min_dist, dist_v))
    {
        std::cerr << "Iteration " << iteration_ << ": distances do not match: " << min_dist << " =/= " << dist_v
                  << std::endl;

        error_ = true;
        return false;
    }

    if (float_diff(clusters_[new_clust].centroid.data(), centroid_v, point_dim_))
    {
        print_arrays(
            iteration_, "centroids do not match", point_dim_, clusters_[new_clust].centroid.data(), centroid_v);

        error_ = true;
        return false;
    }

    std::memcpy(clusters_[new_clust].centroid.data(), centroid_v, point_dim_ * sizeof(float));

    auto this_cov = compute_covariance(clusters_[new_clust]);

    if (float_diff(this_cov, cov_, point_dim_ * point_dim_))
    {
        print_arrays(iteration_, "covariances do not match", point_dim_ * point_dim_, this_cov.data(), cov_.data());

        error_ = true;
        return false;
    }

    check_inverse(clusters_[new_clust]);

    if (apr_sizes_.empty())
        for (size_t i = 0; i < point_count_; i++)
        {
            if (asgns_[i] != point_asgns_[i])
            {
                print_arrays(iteration_, "assignments do not match", point_count_, point_asgns_.data(), asgns_.data());

                error_ = true;
                return false;
            }
        }

    if (error_)
        return false;

    mf_ = 0;
    icmf_ = 0;
    cov_.resize(0);
    icov_.resize(0);
    return true;
}

std::vector<float> validator::compute_covariance(const cluster& c)
{
    auto wf = std::min(1.f, c.size / (float)maha_threshold_);

    if ((subthreshold_kind_ == subthreshold_handling_kind::EUCLID
            || subthreshold_kind_ == subthreshold_handling_kind::EUCLID_MAHAL)
        && wf < 1)
    {
        return {};
    }

    std::vector<float> cov;
    cov.resize(point_dim_ * point_dim_);

    if (c.size == 2)
    {
        cov = unit_matrix_;
        mf_ = 1.f;
        wf = 0;
    }
    else
    {
        for (csize_t i = 0; i < point_dim_; ++i)
            for (csize_t j = i; j < point_dim_; ++j)
            {
                float res = 0;
                csize_t count = 0;
                for (csize_t k = 0; k < point_count_; ++k)
                {
                    if (point_asgns_[k] != c.id)
                        continue;

                    ++count;

                    res +=
                        (points_[k * point_dim_ + i] - c.centroid[i]) * (points_[k * point_dim_ + j] - c.centroid[j]);
                }
                cov[i + point_dim_ * j] = (res / count);
                cov[j + point_dim_ * i] = (res / count);
            }
    }

    if (wf < 1)
    {
        for (csize_t i = 0; i < point_dim_ * point_dim_; i++)
            cov[i] = wf * cov[i] + (1 - wf) * mf_ * unit_matrix_[i];
    }

    return cov;
}

void validator::check_inverse(const cluster& c)
{
    auto wf = std::min(1.f, c.size / (float)maha_threshold_);

    if ((subthreshold_kind_ == subthreshold_handling_kind::EUCLID
            || subthreshold_kind_ == subthreshold_handling_kind::EUCLID_MAHAL)
        && wf < 1)
    {
        if (icmf_ != 1)
        {
            std::cerr << "Iteration " << iteration_ << ": icmf should be 1." << std::endl;

            error_ = true;
            return;
        }
        if (float_diff(icov_, unit_matrix_, point_dim_ * point_dim_))
        {
            print_arrays(
                iteration_, "covariance inverses do not match", point_dim_ * point_dim_, icov_.data(), cov_.data());

            error_ = true;
            return;
        }
    }
    else if (wf != 1)
    {
        if (subthreshold_kind_ == subthreshold_handling_kind::MAHAL0 && mf_ != 1)
        {
            std::cerr << "Iteration " << iteration_ << ": mf should be 1." << std::endl;

            error_ = true;
            return;
        }
    }
}

bool validator::has_error() const { return error_; }

void validator::set_cov(const float* arr)
{
    cov_.resize(point_dim_ * point_dim_);

    CUCH(cudaMemcpy(cov_.data(), arr, sizeof(float) * point_dim_ * point_dim_, cudaMemcpyKind::cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < point_dim_; i++)
        for (size_t j = i + 1; j < point_dim_; j++)
            cov_[i + point_dim_ * j] = cov_[j + point_dim_ * i];
}

void validator::set_icov(const float* arr)
{
    icov_.resize(point_dim_ * point_dim_);

    CUCH(
        cudaMemcpy(icov_.data(), arr, sizeof(float) * point_dim_ * point_dim_, cudaMemcpyKind::cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < point_dim_; i++)
        for (size_t j = i + 1; j < point_dim_; j++)
            icov_[i + point_dim_ * j] = icov_[j + point_dim_ * i];
}

void validator::set_mf(const float* cu_cholesky, const int* cu_info)
{
    int info;
    CUCH(cudaMemcpy(&info, cu_info, sizeof(info), cudaMemcpyKind::cudaMemcpyDeviceToHost));

    if (subthreshold_kind_ == subthreshold_handling_kind::MAHAL0 || info != 0)
    {
        mf_ = 1.f;
        return;
    }

    std::vector<float> cholesky;
    cholesky.resize(point_dim_ * point_dim_);

    CUCH(cudaMemcpy(
        cholesky.data(), cu_cholesky, sizeof(float) * point_dim_ * point_dim_, cudaMemcpyKind::cudaMemcpyDeviceToHost));

    mf_ = 1.f;
    for (csize_t i = 0; i < point_dim_; i++)
        mf_ *= std::pow(cholesky[i * (point_dim_ + 1)], 2.f / point_dim_);
}

void validator::set_icmf(const float* value)
{
    CUCH(cudaMemcpy(&icmf_, value, sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void validator::set_asgns(const asgn_t* value)
{
    if (apr_sizes_.empty())
        CUCH(cudaMemcpy(asgns_.data(), value, sizeof(asgn_t) * point_count_, cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void validator::get_min(const pasgn_t& expected,
    pasgn_t& min_pair,
    std::pair<csize_t, csize_t>& min_idx,
    std::pair<csize_t, csize_t>& expected_idx,
    float& expected_dist,
    float& min_dist)
{
    csize_t from, to;

    while (apr_idx_ != apr_sizes_.size() && apr_sizes_[apr_idx_] == 1)
        ++apr_idx_;

    if (apr_idx_ == apr_sizes_.size())
    {
        from = 0;
        to = cluster_count_;
    }
    else
    {
        from = apr_idx_;
        to = apr_sizes_[apr_idx_] + apr_idx_;
    }

    for (csize_t i = from; i < to; i++)
    {
        for (csize_t j = i + 1; j < to; j++)
        {
            auto l_c = clusters_[i].centroid.data();
            auto r_c = clusters_[j].centroid.data();
            float* l_icov = clusters_[i].icov.data();
            float* r_icov = clusters_[j].icov.data();
            float l_mf = 1, r_mf = 1;

            if (maha_cluster_count_ != cluster_count_)
            {
                if (subthreshold_kind_ == subthreshold_handling_kind::EUCLID)
                {
                    l_icov = unit_matrix_.data();
                    r_icov = unit_matrix_.data();
                }
                else
                {
                    l_mf = clusters_[i].mf;
                    r_mf = clusters_[j].mf;
                }
            }

            auto tmp_dist = compute_distance(l_c, l_icov, l_mf, r_c, r_icov, r_mf, point_dim_);

            pasgn_t curr_ids = clusters_[i].id > clusters_[j].id ? std::make_pair(clusters_[j].id, clusters_[i].id)
                                                                 : std::make_pair(clusters_[i].id, clusters_[j].id);

            if (curr_ids == expected)
            {
                expected_idx = std::make_pair(i, j);
                expected_dist = tmp_dist;
            }

            if (tmp_dist <= min_dist)
            {
                min_pair = curr_ids;
                min_idx = std::make_pair(i, j);
                min_dist = tmp_dist;
            }
        }
    }
}

std::tuple<pasgn_t, csize_t, float> validator::iterate(const pasgnd_t<float>& expected, recompute_f recompute)
{
    pasgn_t v_min_pair;
    std::pair<csize_t, csize_t> v_min_idx, ve_idx;
    float v_min_dist = FLT_INF;
    float ve_dist = FLT_INF;

    get_min(expected.first, v_min_pair, v_min_idx, ve_idx, ve_dist, v_min_dist);

    if (expected.first != v_min_pair) // 0.001f
    {
        float recomputed_dist = 0;
        bool recomputable = expected.first.first != v_min_pair.first && expected.first.second != v_min_pair.second;

        bool good = !float_diff(ve_dist, v_min_dist, 0.001f) && !float_diff(expected.second, v_min_dist, 0.001f);

        if (recomputable)
        {
            recomputed_dist = recompute(v_min_pair);
            good &= recomputed_dist >= expected.second && !float_diff(recomputed_dist, v_min_dist, 0.001f);
        }

        if (good)
        {
            std::cout << "validator branching" << std::endl;
            std::cout << "recomputed: " << recomputable << std::endl;
            std::cout << "expected pair: " << expected.first.first << " " << expected.first.second << std::endl;
            std::cout << "validator min pair: " << v_min_pair.first << " " << v_min_pair.second << std::endl;
            std::cout << "expected dist: " << expected.second << std::endl;
            std::cout << "validator min dist: " << v_min_dist << std::endl;
            if (recomputable)
                std::cout << "recomputed dist: " << recomputed_dist << std::endl;

            v_min_pair = expected.first;
            v_min_dist = expected.second;
            v_min_idx = ve_idx;
        }
    }

    csize_t cluster_size = update_asgns(point_asgns_.data(), point_count_, v_min_pair, id_);

    cluster c;
    c.id = id_;
    c.size = cluster_size;
    c.centroid = compute_centroid(points_, point_dim_, point_count_, point_asgns_.data(), id_);
    c.icov = icov_;
    c.mf = icmf_;

    if (c.size >= maha_threshold_)
    {
        if (clusters_[v_min_idx.first].size >= maha_threshold_ && clusters_[v_min_idx.second].size >= maha_threshold_)
            --maha_cluster_count_;
        else if (clusters_[v_min_idx.first].size < maha_threshold_
            && clusters_[v_min_idx.second].size < maha_threshold_)
            ++maha_cluster_count_;
    }

    clusters_.erase(clusters_.begin() + v_min_idx.second);
    clusters_[v_min_idx.first] = std::move(c);

    ++id_;
    ++iteration_;
    --cluster_count_;

    if (apr_idx_ != apr_sizes_.size())
        --apr_sizes_[apr_idx_];

    return std::tie(v_min_pair, v_min_idx.first, v_min_dist);
}

bool validator::float_diff(float a, float b, float d) { return float_diff(&a, &b, 1, d); }

bool validator::float_diff(const float* a, const float* b, csize_t size, float d)
{
    float fr = 0;
    for (csize_t i = 0; i < size; i++)
    {
        auto diff = std::abs(a[i] - b[i]);
        float tmp;
        if (a[i] == 0 || b[i] == 0)
            tmp = diff;
        else
            tmp = (diff / a[i] + diff / b[i]) / 2;

        fr += tmp;
    }
    if (fr / size >= d)
        return true;
    else
        return false;
}

bool validator::float_diff(const std::vector<float>& a, const std::vector<float>& b, csize_t size, float d)
{
    if (a.size() != b.size())
        return true;
    if (a.empty() && b.empty())
        return false;
    return float_diff(a.data(), b.data(), size, d);
}
