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
    microbench_t,// micro_time,
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
    //{
    //    std::cout << std::get<0>(micro_time) << "\t";
    //    std::cout << std::get<1>(micro_time) << std::endl;
    //}
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

int measure(const float* data, clustering::csize_t count, clustering::csize_t dim, float thresh, size_t repetitions)
{
    using namespace clustering;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "threshold: " << thresh <<  std::endl;
    std::cout << "kind\ttime    \tstd.dev"<< std::endl;//    \tcov      \tnei" << std::endl;

    auto threshold = (csize_t)(count * thresh);

    // dry run
    {
        std::string input =
            "3 32 -1.29547 8.00796 -7.49481 0.873838 0.59632 -1.51012 2.29091 0.944923 1.97487 -11.8492 0.220984 "
            "-3.32246 "
            "-3.18128 -0.856242 8.65632 -3.82426 3.51576 0.726751 0.468351 6.02061 1.30511 -3.4598 8.40714 -3.51013 "
            "0.875993 1.37086 -4.31532 5.93115 6.5827 3.40516 4.71282 4.14184 -2.94859 1.04552 4.30228 1.4721 -2.33436 "
            "-8.34392 -0.658711 7.88312 -3.95856 -4.61158 -5.23649 -2.20981 4.77685 -0.16196 0.051402 4.59879 -10.0206 "
            "0.902125 -0.0203374 1.64312 -2.10842 -1.20475 -12.092 0.0737986 2.18845 -1.17889 -11.8277 9.89593 3.83565 "
            "3.71229 -0.498549 0.436246 1.65099 4.69318 4.50654 4.72271 -3.79102 -5.22239 0.974885 -10.9422 5.59455 "
            "1.47331 5.72225 -1.47315 -2.90234 -7.91345 -9.92892 11.5448 -4.2494 -2.811 1.19261 -5.76345 -0.720766 "
            "16.3369 "
            "4.82052 0.6734 5.29015 1.93098 -2.19635 -2.60147 -3.42446 1.9895 -1.261 -2.28809";

        auto dry_data = reader::read_data_from_string<float>(input);

        gmhc gmhclust;
        if (!gmhclust.initialize(dry_data.data.data(),
                (csize_t)dry_data.points,
                (csize_t)dry_data.dim,
                (csize_t)5,
                subthreshold_handling_kind::MAHAL))
            return 1;
        gmhclust.run();
        gmhclust.free();
    }

    gmhc gmhclust;

    std::vector<std::chrono::duration<double>> run_time;
    std::vector<float> cov_time;
    std::vector<float> nei_time;
    microbench_t micro_res;

    std::vector<double> processed_time;
    size_t enum_size = 3;

    for (size_t i = 0; i < enum_size; i++)
    {
        auto kind = (subthreshold_handling_kind)i;

        for (size_t j = 0; j < repetitions; j++)
        {
            // gmhclust.timer().initialize();

            auto start = std::chrono::system_clock::now();

            if (!gmhclust.initialize(data, count, dim, threshold, kind))
                return 1;
            gmhclust.run();
            gmhclust.free();

            auto end = std::chrono::system_clock::now();

            run_time.push_back(end - start);

            if (j == repetitions - 1)
                micro_res = compute_micro_time(gmhclust.timer());
            // gmhclust.timer().free();
        }
        print_time(run_time, micro_res, kind);
        std::cout << thresh << std::endl;
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
    cmd.add_option("t", "Mahalanobis threshold.", false)
        .add_parameter<float>(value_type<float>(), "THRESH")
        .set_constraint([](float value) { return value <= 1 && value >= 0; });

    auto parsed = cmd.parse(argc, argv);

    if (!parsed.parse_ok())
    {
        std::cerr << cmd.help();
        return 1;
    }

    auto dataset = parsed["dataset"]->get_value<std::string>();
    auto data = reader::read_data_from_binary_file<float, double>(dataset);
    auto reps = parsed["n"]->get_value<int>();
    auto thresh = parsed["t"]->get_value<float>();

    return measure(data.data.data(), (csize_t)data.points, (csize_t)data.dim, thresh, (size_t)reps);
}
