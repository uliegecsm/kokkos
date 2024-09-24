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

namespace {

// Check Kokkos::Impl::is_view_label.
TEST(TEST_CATEGORY, is_view_label) {
  static_assert(Kokkos::Impl::is_view_label<std::string>::value);

  constexpr unsigned N = 3;
  static_assert(Kokkos::Impl::is_view_label<const char[N]>::value);
  static_assert(Kokkos::Impl::is_view_label<char[N]>::value);

  // A char* is not a label. Thus, a label is distinguished from a pointer type.
  static_assert(!Kokkos::Impl::is_view_label<char*>::value);
}

// Check traits of Kokkos::Impl::ViewCtorProp<>.
TEST(TEST_CATEGORY, vcp_empty_traits) {
  using vcp_empty_t = Kokkos::Impl::ViewCtorProp<>;

  // Check that the empty view constructor properties class is default
  // constructible. This is needed for calls of Kokkos::view_alloc().
  static_assert(std::is_default_constructible_v<vcp_empty_t>);

  static_assert(std::is_same_v<decltype(Kokkos::view_alloc()), vcp_empty_t>);
}

// Check traits of base class Kokkos::Impl::ViewCtorProp<void, std::string>.
TEST(TEST_CATEGORY, vcp_label_base_traits) {
  using vcp_label_base_t = Kokkos::Impl::ViewCtorProp<void, std::string>;

  static_assert(std::is_same_v<typename vcp_label_base_t::type, std::string>);

  // Check that the base class is default constructible. The default constructor
  // may be called by the copy constructor of deriveded classes, such as when
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

// Check traits of derived class Kokkos::Impl::ViewCtorProp<std::string>.
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

  static_assert(std::is_constructible_v<vcp_label_t, char*>);
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

// Check the copy constructor of Kokkos::Impl::ViewCtorProp<std::string>.
TEST(TEST_CATEGORY, vcp_label_copy_constructor) {
  using vcp_empty_t = Kokkos::Impl::ViewCtorProp<>;
  using vcp_label_t = Kokkos::Impl::ViewCtorProp<std::string>;

  // Copy construction from an empty view constructor properties object.
  static_assert(std::is_constructible_v<vcp_label_t, const vcp_empty_t&>);

  vcp_empty_t prop_empty;
  vcp_label_t prop_empty_copy(prop_empty);

  ASSERT_TRUE(
      Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop_empty_copy)
          .empty());

  // Copy construction from a view constructor properties object with a label.
  static_assert(std::is_copy_constructible_v<vcp_label_t>);

  auto prop = Kokkos::view_alloc("our label");
  vcp_label_t prop_copy(prop);

  ASSERT_EQ(Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop),
            "our label");
  ASSERT_EQ(Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop_copy),
            "our label");
}

}  // namespace
