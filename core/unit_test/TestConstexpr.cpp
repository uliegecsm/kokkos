#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

constexpr size_t size = 5;

using device_type     = Kokkos::View<int*>::device_type;
using execution_space = typename device_type::execution_space;
using view_t          = Kokkos::View<double[size], execution_space>;

static const view_t my_view; // needed otherwise I get incomplete type for ViewTracker... wtf

using view_tracker_t  = Kokkos::Impl::ViewTracker<view_t>;
using track_t         = typename view_tracker_t::track_type;
using map_t           = Kokkos::Impl::ViewMapping<typename view_t::traits, typename view_t::traits::specialize>;

TEST(Constexprness, SharedAllocationTracker)
{
    static_assert([](){
        track_t track;
        // do something with it ?
    });
}

TEST(Constexpr, ViewTracker)
{
    static_assert([](){
        view_tracker_t view_tracker;
        // do something with it ?
    });
}

TEST(Constexpr, ViewMapping)
{
    static_assert([](){
        map_t map;
        // do somthing meaningful with it ?
        return map.m_impl_handle == nullptr;
    });
}


TEST(KokkosView, contructor_constexprness)
{
    static_assert([](){
        view_t my_view;
        return my_view.size() == my_view.extent(0) && my_view.size() == size;
    });
}
