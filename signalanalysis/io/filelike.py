#
# This file is part of openCARP
# (see https://www.openCARP.org).
#
# The openCARP project licenses this file to you under the 
# Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

class FileLikeMixin(object):
    """
    Implements a file-like mixin that essentially allows self-closing when exiting a "with-block".

    For information on context managers see `https://docs.python.org/2/library/stdtypes.html#context-manager-types`
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the file
        self.close()
        # Do not suppress any exceptions
        return False
