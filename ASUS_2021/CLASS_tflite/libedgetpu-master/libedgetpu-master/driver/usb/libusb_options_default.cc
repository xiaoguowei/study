// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "driver/usb/libusb_options.h"

namespace platforms {
namespace darwinn {
namespace driver {

// The implementation of SetLibUsbOptions for the default platform
// is a no-op. On other platforms, this may do something of interest.
int SetLibUsbOptions(libusb_context* context) { return LIBUSB_SUCCESS; }

}  // namespace driver
}  // namespace darwinn
}  // namespace platforms
