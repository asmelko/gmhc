#ifndef GMHC_HPP
#define GMHC_HPP

#include <clustering.hpp>

namespace clustering
{

class gmhc : public hierarchical_clustering<float>
{
public:
    virtual std::vector<pasgn_t> run() override;
    virtual void free() override;
};

}

#endif