#ifndef PANOPTIC_MAPPING_ROS_UTILS_MKDIR_P_H_
#define PANOPTIC_MAPPING_ROS_UTILS_MKDIR_P_H_
#include <errno.h>
#include <limits.h> /* PATH_MAX */
#include <string.h>
#include <sys/stat.h> /* mkdir(2) */

inline int mkdir_p(const char* path) {
  // Adapted from http://stackoverflow.com/a/2336245/119527
  const size_t len = strlen(path);
  char _path[PATH_MAX];
  char* p;

  errno = 0;

  /* Copy string so its mutable */
  if (len > sizeof(_path) - 1) {
    errno = ENAMETOOLONG;
    return -1;
  }
  snprintf(_path, path);

  /* Iterate the string */
  for (p = _path + 1; *p; p++) {
    if (*p == '/') {
      /* Temporarily truncate */
      *p = '\0';

      if (mkdir(_path, S_IRWXU) != 0) {
        if (errno != EEXIST) return -1;
      }

      *p = '/';
    }
  }

  if (mkdir(_path, S_IRWXU) != 0) {
    if (errno != EEXIST) return -1;
  }

  return 0;
}

#endif  // PANOPTIC_MAPPING_ROS_UTILS_MKDIR_P_H_
