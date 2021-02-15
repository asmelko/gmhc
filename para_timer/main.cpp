#include <chrono>
#include <climits>
#include <iomanip>
#include <iostream>
#include <istream>
#include <numeric>
#include <tuple>

#include "../para/include/gmhc.hpp"
#include "option.hpp"
#include "reader.hpp"

std::string enum_to_string(clustering::subthreshold_handling_kind kind)
{
    switch (kind)
    {
        case clustering::subthreshold_handling_kind::MAHAL:
            return "M";
        case clustering::subthreshold_handling_kind::EUCLID:
            return "E";
        case clustering::subthreshold_handling_kind::MAHAL0:
            return "M0";
        case clustering::subthreshold_handling_kind::EUCLID_MAHAL:
            return "EM";
        default:
            return "UNDEF";
    }
}

using microbench_t = std::tuple<float, float>;

void print_time(std::vector<std::chrono::duration<double>>& time,
    microbench_t micro_time,
    clustering::subthreshold_handling_kind kind)
{
    std::cout << enum_to_string(kind) << "\t";

    {
        auto sum = std::accumulate(time.begin(), time.end(), std::chrono::duration<double>::zero());
        auto mean = sum / time.size();
        std::cout << mean.count() << "\t";

        double std_dev = 0;
        for (auto t : time)
            std_dev += (t - mean).count() * (t - mean).count();
        std_dev /= time.size();
        std_dev = std::sqrt(std_dev);
        std::cout << std_dev << "\t";
        time.clear();
    }
    {
        std::cout << std::get<0>(micro_time) << "\t";
        std::cout << std::get<1>(micro_time) << std::endl;
    }
}

microbench_t compute_micro_time(const clustering::time_info& timer)
{
    microbench_t t;
    {
        std::get<0>(t) = std::accumulate(timer.cov_time.begin(), timer.cov_time.end(), 0.f) / 1000;
    }
    {
        std::get<1>(t) = std::accumulate(timer.nei_time.begin(), timer.nei_time.end(), 0.f) / 1000;
    }
    return t;
}

template <clustering::csize_t Neighbors>
int measure(const float* data, clustering::csize_t count, clustering::csize_t dim, size_t repetitions)
{
    using namespace clustering;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "kind\ttime    \tstd.dev    \tcov      \tnei" << std::endl;

    auto threshold = (csize_t)(count * 0.5);

    // dry run
    {
        gmhc<Neighbors> gmhclust;
        if (!gmhclust.initialize(data, count, dim, threshold, subthreshold_handling_kind::MAHAL))
            return 1;
        gmhclust.run();
        gmhclust.free();
    }

    gmhc<Neighbors> gmhclust;

    std::vector<std::chrono::duration<double>> run_time;
    std::vector<float> cov_time;
    std::vector<float> nei_time;
    microbench_t micro_res;

    std::vector<double> processed_time;
    size_t enum_size = 4;

    for (size_t i = 0; i < enum_size; i++)
    {
        auto kind = (subthreshold_handling_kind)i;

        for (size_t j = 0; j < repetitions; j++)
        {
            gmhclust.timer().initialize();

            auto start = std::chrono::system_clock::now();

            if (!gmhclust.initialize(data, count, dim, threshold, kind))
                return 1;
            gmhclust.run();
            gmhclust.free();

            auto end = std::chrono::system_clock::now();

            run_time.push_back(end - start);

            if (j == repetitions - 1)
                micro_res = compute_micro_time(gmhclust.timer());
            gmhclust.timer().free();
        }
        print_time(run_time, micro_res, kind);
    }

    return 0;
}

int main(int argc, char** argv)
{
    using namespace mbas;
    using namespace clustering;

    command cmd;

    cmd.add_option("d,dataset", "Path to dataset file.", false)
        .add_parameter<std::string>(value_type<std::string>(), "D_PATH");
    cmd.add_option("n", "Test repetition.", false)
        .add_parameter<int>(value_type<int>(), "REPS")
        .set_constraint([](int value) { return value >= 1; });
    cmd.add_option("N", "Number of neighbors in the closest neighbor array.", false)
        .add_parameter<int>(value_type<int>(), "NEIGHS")
        .set_constraint([](int value) { return value >= 1; });

    auto parsed = cmd.parse(argc, argv);

    if (!parsed.parse_ok())
    {
        std::cerr << cmd.help();
        return 1;
    }

    auto dataset = parsed["dataset"]->get_value<std::string>();
    auto data = reader::read_data_from_binary_file<float, double>(dataset);
    auto reps = parsed["n"]->get_value<int>();
    auto neighs = parsed["N"]->get_value<int>();

    switch (neighs)
    {
        case 1:
            return measure<1>(data.data.data(), (csize_t)data.points, (csize_t)data.dim, (size_t)reps);
        case 2:
            return measure<2>(data.data.data(), (csize_t)data.points, (csize_t)data.dim, (size_t)reps);
        case 3:
            return measure<3>(data.data.data(), (csize_t)data.points, (csize_t)data.dim, (size_t)reps);
        case 4:
            return measure<4>(data.data.data(), (csize_t)data.points, (csize_t)data.dim, (size_t)reps);
        case 5:
            return measure<5>(data.data.data(), (csize_t)data.points, (csize_t)data.dim, (size_t)reps);
        default:
            return measure<1>(data.data.data(), (csize_t)data.points, (csize_t)data.dim, (size_t)reps);
    }
}
