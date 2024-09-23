//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace Test {

// Check Kokkos::Impl::is_view_label.
TEST(TEST_CATEGORY, is_view_label) {
  static_assert(Kokkos::Impl::is_view_label<std::string>::value);

  constexpr unsigned N = 3;
  static_assert(Kokkos::Impl::is_view_label<const char[N]>::value);
  static_assert(Kokkos::Impl::is_view_label<char[N]>::value);

  static_assert(!Kokkos::Impl::is_view_label<char*>::value);
}

// Check traits of Kokkos::Impl::ViewCtorProp<void, std::string>.
TEST(TEST_CATEGORY, vcp_label_base_traits) {
  using vcp_label_base_t = Kokkos::Impl::ViewCtorProp<void, std::string>;

  static_assert(std::is_same_v<typename vcp_label_base_t::type, std::string>);

  // Check that the base class is default constructible. The default constructor
  // may be called by the copy constructor of the derived class, such as when
  // copy constructing a view constructor properties object from another view
  // constructor properties object that holds fewer properties.
  static_assert(std::is_default_constructible_v<vcp_label_base_t>);

  static_assert(std::is_constructible_v<vcp_label_base_t, std::string>);
  static_assert(std::is_constructible_v<vcp_label_base_t, const std::string&>);
  static_assert(std::is_constructible_v<vcp_label_base_t, std::string&&>);

  constexpr unsigned N = 3;
  static_assert(std::is_constructible_v<vcp_label_base_t, const char[N]>);
  static_assert(std::is_constructible_v<vcp_label_base_t, char[N]>);

  static_assert(std::is_constructible_v<vcp_label_base_t, char*>);
}

// Check traits of Kokkos::Impl::ViewCtorProp<std::string>.
TEST(TEST_CATEGORY, vcp_label_traits) {
  using vcp_label_base_t = Kokkos::Impl::ViewCtorProp<void, std::string>;
  using vcp_label_t      = Kokkos::Impl::ViewCtorProp<std::string>;

  static_assert(std::is_base_of_v<vcp_label_base_t, vcp_label_t>);

  static_assert(vcp_label_t::has_label);

  // Check that the derived class is not default constructible. It is a design
  // choice to not allow the default constructor to be called.
  static_assert(!std::is_default_constructible_v<vcp_label_t>);

  static_assert(std::is_constructible_v<vcp_label_t, std::string>);
  static_assert(std::is_constructible_v<vcp_label_t, const std::string&>);
  static_assert(std::is_constructible_v<vcp_label_t, std::string&&>);

  constexpr unsigned N = 3;
  static_assert(std::is_constructible_v<vcp_label_t, const char[N]>);
  static_assert(std::is_constructible_v<vcp_label_t, char[N]>);

  // Kokkos::Impl::ViewCtorProp<std::string> cannot be constructed from
  // a `char*` because `char*` does not satisfy `Kokkos::Impl::is_view_label`,
  // hence the constructor cannot access the type alias `type`.
  static_assert(!std::is_constructible_v<vcp_label_t, char*>);
}

// Check that the constructor of Kokkos::Impl::ViewCtorProp<std::string>
// moves a label passed by rvalue reference.
TEST(TEST_CATEGORY, vcp_label_constructor_can_move) {
  using vcp_label_t = Kokkos::Impl::ViewCtorProp<std::string>;

  std::string label("our label");

  vcp_label_t prop(std::move(label));

  ASSERT_TRUE(label.empty());
  ASSERT_EQ(Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop),
            "our label");
}

// Check that Kokkos::view_alloc moves a label passed by rvalue reference.
TEST(TEST_CATEGORY, vcp_label_view_alloc_can_move) {
  std::string label("our label");

  auto prop = Kokkos::view_alloc(std::move(label));

  ASSERT_TRUE(label.empty());
  ASSERT_EQ(Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop),
            "our label");
}

}  // namespace Test
